import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import os
from pathlib import Path
import torch
import argparse
from tqdm import tqdm
import logging
from typing import Optional, Tuple, List, Dict, Any

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_device(force_cpu: bool = False) -> torch.device:
    """Get the appropriate device (CUDA if available and not forced to CPU, CPU otherwise)."""
    if force_cpu:
        logging.info("Forcing CPU usage")
        return torch.device('cpu')
    elif torch.cuda.is_available():
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        logging.warning("CUDA not available, using CPU")
        return torch.device('cpu')

def calculate_parameter_count(shape: Tuple[int, ...], rank: int) -> int:
    """Calculate the number of parameters in the PARAFAC decomposition."""
    return sum(shape[i] * rank for i in range(len(shape)))

def calculate_reconstruction_error(
    tensor: torch.Tensor,
    weights: torch.Tensor,
    factors: List[torch.Tensor],
    device: torch.device
) -> Tuple[float, float, float, float]:
    """Calculate error metrics for tensor reconstruction.
    
    Returns:
        Tuple containing:
        - misclassification: Rate of incorrect binary classifications
        - f1_error: 1 - F1 score for binary classification
        - mse: Mean squared error of the reconstruction
        - rel_error: Relative reconstruction error (Frobenius norm of difference / Frobenius norm of original)
    """
    reconstructed = tl.cp_to_tensor((weights, factors))
    
    # Calculate binary classification metrics
    binary_reconstruction = (reconstructed > 0.5).to(tensor.dtype)
    misclassification = torch.mean((binary_reconstruction != tensor).float())
    
    true_positives = torch.sum((binary_reconstruction == 1) & (tensor == 1))
    precision = true_positives / (torch.sum(binary_reconstruction == 1) + 1e-10)
    recall = true_positives / (torch.sum(tensor == 1) + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Calculate total reconstruction error
    mse = torch.mean((reconstructed - tensor) ** 2)
    rel_error = torch.norm(reconstructed - tensor) / torch.norm(tensor)
    
    return misclassification.item(), 1 - f1_score.item(), mse.item(), rel_error.item()

def find_optimal_rank(
    tensor: torch.Tensor,
    max_rank: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> Tuple[int, float, float, float, float]:
    """Find rank that gives the best reconstruction quality using exponential interpolation."""
    shape = tensor.shape
    best_rank = None
    best_misclassification = float('inf')
    best_f1_error = float('inf')
    best_mse = float('inf')
    best_rel_error = float('inf')
    
    # Convert tensor to appropriate device and dtype if using CUDA
    if device.type == 'cuda':
        tensor = tensor.to(device=device, dtype=torch.float32)
    
    # Generate exponentially spaced ranks
    if max_rank <= 10:
        ranks = list(range(1, max_rank + 1))
    else:
        # Use specific ranks: 1, 20, 200, 1000
        ranks = [1, 20, 200, 1000, 1500]
    
    if verbose:
        logging.info(f"Testing ranks: {ranks}")
    
    pbar = tqdm(ranks, desc="Finding optimal rank", disable=not verbose)
    
    for current_rank in pbar:
        try:
            init = 'svd' if tensor.numel() < 900000 else 'random'
            # Use SVD initialization and L2 regularization
            weights, factors = parafac(tensor, rank=current_rank, 
                                    init=init,  # More stable initialization
                                    svd='randomized_svd',  # Faster SVD variant
                                    tol=1e-6,
                                    n_iter_max=100,
                                    l2_reg=1e-6)  # Small L2 regularization
            
            misclassification, f1_error, mse, rel_error = calculate_reconstruction_error(tensor, weights, factors, device)
            
            # Update best if either misclassification or relative error improved
            if misclassification < best_misclassification or rel_error < best_rel_error:
                best_rank = current_rank
                best_misclassification = misclassification
                best_f1_error = f1_error
                best_mse = mse
                best_rel_error = rel_error
                if verbose:
                    logging.debug(f"Found better rank {current_rank}:")
                    logging.debug(f"  Misclassification: {misclassification:.4f}")
                    logging.debug(f"  F1 error: {f1_error:.4f}")
                    logging.debug(f"  MSE: {mse:.4e}")
                    logging.debug(f"  Relative error: {rel_error:.4e}")
            
            if misclassification == 0 and rel_error < 1e-3:  # Only stop if both are very good
                if verbose:
                    logging.info("Perfect reconstruction achieved, stopping.")
                break
            
            # Clean up GPU memory
            if device.type == 'cuda':
                del weights
                del factors
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.warning(f"Error during decomposition with rank {current_rank}: {str(e)}")
            if best_rank is not None:
                return best_rank, best_misclassification, best_f1_error, best_mse, best_rel_error
    
    if best_rank is None:
        best_rank = 1
    
    return best_rank, best_misclassification, best_f1_error, best_mse, best_rel_error

def decompose_tensor(
    tensor: torch.Tensor,
    output_dir: Path,
    tensor_name: str,
    rank: int = 1000,
    verbose: bool = False,
    device: torch.device = None
) -> Dict[str, Any]:
    """Decompose a single tensor and save its components."""
    logging.info(f"Processing tensor: {tensor_name}")
    
    # Set appropriate backend
    tl.set_backend('pytorch')
    
    # Move tensor to appropriate device and convert to 32 bit if using CUDA
    if device.type == 'cuda':
        tensor = tensor.to(device=device, dtype=torch.float32)
    
    init = 'svd' if tensor.numel() < 900000 else 'random'
    if verbose:
        logging.info(f"Using {init} initialization")
    # Perform decomposition with more stable parameters
    weights, factors = parafac(tensor, rank=rank, 
                            init=init,  # More stable initialization
                            svd='randomized_svd',  # Faster SVD variant
                            tol=1e-8,
                            n_iter_max=300,
                            l2_reg=1e-6,  # Small L2 regularization
                            verbose=verbose)

    if verbose:
        logging.debug(f"Tensor decomposition complete")
    
    final_misclassification, final_f1_error, final_mse, final_rel_error = calculate_reconstruction_error(tensor, weights, factors, device)
    
    logging.info(f"Tensor shape: {tensor.shape}")
    logging.info(f"Rank: {rank}")
    logging.info(f"Misclassification rate: {final_misclassification:.4%}")
    logging.info(f"F1 error: {final_f1_error:.4%}")
    logging.info(f"MSE: {final_mse:.4e}")
    logging.info(f"Relative error: {final_rel_error:.4e}")
    
    # Convert results to numpy for saving with explicit dtypes
    weights_np = weights.cpu().numpy().astype(np.float32)
    factors_np = [f.cpu().numpy().astype(np.float32) for f in factors]
    
    # Clean up GPU memory
    if device.type == 'cuda':
        del weights
        del factors
        torch.cuda.empty_cache()
    
    # Save results
    np.save(output_dir / "weights.npy", weights_np)
    for i, factor in enumerate(factors_np):
        np.save(output_dir / f"factor_{i}.npy", factor)
    
    # Save metadata
    metadata = {
        'shape': tensor.shape,
        'rank': rank,
        'misclassification': float(final_misclassification),
        'f1_error': float(final_f1_error),
        'mse': float(final_mse),
        'relative_error': float(final_rel_error),
        'device_used': device.type
    }
    np.save(output_dir / "metadata.npy", metadata)
    
    logging.info(f"Saved decomposition for {tensor_name}")
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Perform PARAFAC decomposition on tensors')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--rank', type=int, default=1000, help='Rank for PARAFAC decomposition')
    parser.add_argument('--find-rank', action='store_true', help='Find optimal rank instead of using fixed rank')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    device = get_device(force_cpu=args.cpu)
    tl.set_backend('pytorch')
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/parafac_decomposed_tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the tensors from raw_tensors
    input_path = Path("data/raw_tensors")
    
    # Process each grid size directory
    grid_dirs = list(input_path.iterdir())
    for grid_dir in tqdm(grid_dirs, desc="Processing grid sizes", disable=not args.verbose):
        if not grid_dir.is_dir():
            continue
            
        logging.info(f"Processing grid size directory: {grid_dir.name}")
        
        # Create corresponding output directory
        grid_output_dir = output_dir / grid_dir.name
        grid_output_dir.mkdir(exist_ok=True)
        
        # Process each tensor in the grid size directory
        tensor_files = list(grid_dir.glob("*.npy"))
        for tensor_file in tqdm(tensor_files, desc=f"Processing tensors in {grid_dir.name}", 
                              disable=not args.verbose, leave=False):
            tensor_dir = grid_output_dir / tensor_file.stem
            tensor_dir.mkdir(exist_ok=True)
            
            # Load tensor directly to the right device and type
            if device.type == 'cuda':
                tensor = torch.from_numpy(np.load(tensor_file)).to(device=device, dtype=torch.float32)
            else:
                tensor = torch.from_numpy(np.load(tensor_file))
        
            
            # Find optimal rank if requested
            if args.find_rank:
                optimal_rank, misclassification, f1_error, mse, rel_error = find_optimal_rank(
                    tensor, 
                    max_rank=args.rank,
                    device=device,
                    verbose=args.verbose
                )
                logging.info(f"Found optimal rank: {optimal_rank}")
                rank = optimal_rank
            else:
                rank = args.rank
            
            # Decompose and save
            decompose_tensor(tensor, tensor_dir, tensor_file.name, rank=rank, verbose=args.verbose, device=device)
            
            # Clear the tensor from memory
            del tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        logging.info(f"Completed processing {grid_dir.name}")

if __name__ == "__main__":
    main() 