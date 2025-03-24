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
) -> Tuple[float, float]:
    """Calculate error metrics appropriate for binary tensors."""
    reconstructed = tl.cp_to_tensor((weights, factors))
    binary_reconstruction = (reconstructed > 0.5).to(tensor.dtype)
    misclassification = torch.mean((binary_reconstruction != tensor).float())
    
    true_positives = torch.sum((binary_reconstruction == 1) & (tensor == 1))
    precision = true_positives / (torch.sum(binary_reconstruction == 1) + 1e-10)
    recall = true_positives / (torch.sum(tensor == 1) + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return misclassification.item(), 1 - f1_score.item()

def find_optimal_rank(
    tensor: torch.Tensor,
    max_rank: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> Tuple[int, float, float]:
    """Find rank that gives the best reconstruction quality using exponential interpolation."""
    shape = tensor.shape
    best_rank = None
    best_misclassification = float('inf')
    best_f1_error = float('inf')
    
    # Convert tensor to appropriate device and dtype if using CUDA
    if device.type == 'cuda':
        tensor = tensor.to(device=device, dtype=torch.float32)
    
    # Generate exponentially spaced ranks
    if max_rank <= 10:
        ranks = list(range(1, max_rank + 1))
    else:
        # Use specific ranks: 1, 20, 200, 1000
        ranks = [1, 20, 200, 500, 1000, 1500]
    
    if verbose:
        logging.info(f"Testing ranks: {ranks}")
    
    pbar = tqdm(ranks, desc="Finding optimal rank", disable=not verbose)
    
    for current_rank in pbar:
        try:
            weights, factors = parafac(tensor, rank=current_rank, 
                                    init='random',
                                    tol=1e-6,
                                    n_iter_max=100)
            
            misclassification, f1_error = calculate_reconstruction_error(tensor, weights, factors, device)
            
            if misclassification < best_misclassification:
                best_rank = current_rank
                best_misclassification = misclassification
                best_f1_error = f1_error
                if verbose:
                    logging.debug(f"Found better rank {current_rank}: misclassification={misclassification:.4f}, F1_error={f1_error:.4f}")
            
            if misclassification == 0 and f1_error == 0:
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
                return best_rank, best_misclassification, best_f1_error
    
    if best_rank is None:
        best_rank = 1
    
    return best_rank, best_misclassification, best_f1_error

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
    
    # Perform decomposition
    weights, factors = parafac(tensor, rank=rank, 
                              init='random',
                              tol=1e-6,
                              n_iter_max=150)
    
    if verbose:
        logging.debug(f"Tensor decomposition complete")
    
    final_misclassification, final_f1_error = calculate_reconstruction_error(tensor, weights, factors, device)
    
    logging.info(f"Tensor shape: {tensor.shape}")
    logging.info(f"Rank: {rank}")
    logging.info(f"Misclassification rate: {final_misclassification:.4%}")
    logging.info(f"F1 error: {final_f1_error:.4%}")
    
    # Convert results to numpy for saving (convert back to float32 for stability)
    weights_np = weights.cpu().float().numpy()
    factors_np = [f.cpu().float().numpy() for f in factors]
    
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
    
    # Get appropriate device
    device = get_device(force_cpu=args.cpu)
    
    # Set the backend to PyTorch
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
            # Create a separate directory for this tensor's decomposition
            tensor_dir = grid_output_dir / tensor_file.stem
            tensor_dir.mkdir(exist_ok=True)
            
            # Load tensor and convert to PyTorch
            tensor = torch.from_numpy(np.load(tensor_file))
            
            # Find optimal rank if requested
            if args.find_rank:
                optimal_rank, misclassification, f1_error = find_optimal_rank(
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
            
        logging.info(f"Completed processing {grid_dir.name}")

if __name__ == "__main__":
    main() 