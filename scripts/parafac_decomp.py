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
import datetime

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration to write to a file."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"parafac_decomp_{timestamp}.log"
    
    # Configure logging to write to both file and terminal
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Keep terminal output as well
        ]
    )
    logging.info(f"Logging to file: {log_file}")

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

def load_tensor_safely(file_path: Path) -> np.ndarray:
    """Load tensor data from a .npy file with special handling for BitArrays from Julia.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        Loaded tensor as a numpy array with proper dtype
    """
    try:
        # Load the raw data
        data = np.load(file_path)
        
        # Print detailed info about the loaded tensor
        logging.info(f"Loaded tensor from {file_path.name}:")
        logging.info(f"  - Shape: {data.shape}")
        logging.info(f"  - Dtype: {data.dtype}")
        
        # Check unique values to identify potential BitArray
        unique_vals = np.unique(data)
        logging.info(f"  - Unique values: {unique_vals[:10]}... (total: {len(unique_vals)})")
        
        # Always convert to float32, regardless of original type
        # This helps with BitArrays and other non-standard numerical types from Julia
        data = data.astype(np.float32)
        logging.info(f"  - Converted to dtype: {data.dtype}")
        
        # Try to identify the reason for NaNs
        if np.isnan(data).any() or np.isinf(data).any():
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            logging.warning(f"  - Found {nan_count} NaN and {inf_count} Inf values")
            
            # Print locations of some NaNs to help debug
            if nan_count > 0:
                nan_indices = np.where(np.isnan(data))
                sample_indices = [(dim[i] for dim in nan_indices) for i in range(min(5, len(nan_indices[0])))]
                logging.warning(f"  - Sample NaN positions: {sample_indices}")
                
                # Check values around the NaNs to look for patterns
                for pos in sample_indices[:3]:  # Look at up to 3 NaN positions
                    pos_tuple = tuple(pos)
                    try:
                        # Create slices to get adjacent values
                        slices = []
                        for i, idx in enumerate(pos_tuple):
                            if idx > 0:
                                pos_before = list(pos_tuple)
                                pos_before[i] = idx - 1
                                slices.append(tuple(pos_before))
                            if idx < data.shape[i] - 1:
                                pos_after = list(pos_tuple)
                                pos_after[i] = idx + 1
                                slices.append(tuple(pos_after))
                                
                        adjacent_vals = [data[s] for s in slices]
                        logging.warning(f"  - Values adjacent to NaN at {pos_tuple}: {adjacent_vals}")
                    except Exception as e:
                        logging.warning(f"  - Error examining adjacent values: {str(e)}")
            
            # Replace with zeros - NaNs will cause SVD to fail
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            logging.info("  - Fixed NaN/Inf values by replacing with zeros")
            
        return data
    
    except Exception as e:
        logging.error(f"Error loading tensor from {file_path}: {str(e)}")
        raise

def check_tensor_for_issues(tensor: torch.Tensor, tensor_name: str) -> bool:
    """Check a tensor for NaN or Inf values and print diagnostics if found.
    
    Args:
        tensor: Tensor to check
        tensor_name: Name of the tensor for reporting
        
    Returns:
        True if the tensor has issues, False otherwise
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        # Count NaN and Inf values
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        
        # Calculate percentage of problematic values
        problem_percentage = (nan_count + inf_count) / total_elements * 100
        
        logging.error(f"❌ TENSOR ISSUE DETECTED in {tensor_name}:")
        logging.error(f"  - Shape: {tensor.shape}")
        logging.error(f"  - NaN values: {nan_count} ({nan_count/total_elements:.6%})")
        logging.error(f"  - Inf values: {inf_count} ({inf_count/total_elements:.6%})")
        logging.error(f"  - Total problematic: {nan_count + inf_count} of {total_elements} elements ({problem_percentage:.6%})")
        
        # Find positions of some NaN values for debugging
        if has_nan:
            nan_indices = torch.where(torch.isnan(tensor))
            sample_indices = [(dim[i].item() for dim in nan_indices) for i in range(min(5, len(nan_indices[0])))]
            logging.error(f"  - Sample NaN positions: {sample_indices}")
        
        return True
    
    return False

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
    
    # Check tensor for NaN/Inf values before decomposition
    if check_tensor_for_issues(tensor, "input"):
        logging.warning("Fixing tensor before decomposition")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # More aggressive NaN/Inf cleaning
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logging.warning(f"Found NaN/Inf values in {tensor_name}, cleaning...")
        # Replace with zeros and log statistics
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        logging.warning(f"  - NaN values: {nan_count} ({nan_count/total_elements:.6%})")
        logging.warning(f"  - Inf values: {inf_count} ({inf_count/total_elements:.6%})")
        
        # Replace with zeros
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify cleaning worked
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logging.error(f"❌ Failed to clean all NaN/Inf values in {tensor_name}")
            return {"error": "Failed to clean NaN/Inf values", "had_nan_inf": True}
    
    init = 'svd' if tensor.numel() < 900000 else 'random'
    if verbose:
        logging.info(f"Using {init} initialization")
    
    try:
        # Try decomposition with current device
        try:
            weights, factors = parafac(tensor, rank=rank, 
                                    init=init,
                                    svd='randomized_svd',
                                    tol=1e-8,
                                    n_iter_max=300,
                                    l2_reg=1e-6,
                                    verbose=verbose)
        except Exception as e:
            # If CUDA failed, try CPU fallback
            if device.type == 'cuda' and ("cusolver error" in str(e) or "NaN" in str(e)):
                logging.warning(f"⚠️ CUDA decomposition failed for {tensor_name}, falling back to CPU")
                # Move tensor to CPU
                tensor_cpu = tensor.cpu()
                
                # Try decomposition on CPU
                weights, factors = parafac(tensor_cpu, rank=rank,
                                        init=init,
                                        svd='randomized_svd',
                                        tol=1e-8,
                                        n_iter_max=300,
                                        l2_reg=1e-6,
                                        verbose=verbose)
                
                # Move results back to original device
                weights = weights.to(device)
                factors = [f.to(device) for f in factors]
            else:
                raise  # Re-raise if not a CUDA error or if fallback failed

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
            'device_used': device.type,
            'had_nan_inf': torch.isnan(tensor).any() or torch.isinf(tensor).any()
        }
        np.save(output_dir / "metadata.npy", metadata)
        
        logging.info(f"Saved decomposition for {tensor_name}")
        return metadata
    except Exception as e:
        logging.error(f"❌ Error during decomposition of {tensor_name}: {str(e)}")
        return {"error": str(e), "had_nan_inf": torch.isnan(tensor).any() or torch.isinf(tensor).any()}

def main():
    parser = argparse.ArgumentParser(description='Perform PARAFAC decomposition on tensors')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--rank', type=int, default=1000, help='Rank for PARAFAC decomposition')
    parser.add_argument('--find-rank', action='store_true', help='Find optimal rank instead of using fixed rank')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    parser.add_argument('--skip-errors', action='store_true', help='Skip tensors that cause errors instead of stopping')
    parser.add_argument('--fix-bit-arrays', action='store_true', help='Apply special handling for BitArrays from Julia', default=True)
    parser.add_argument('--fall-back-to-cpu', action='store_true', help='Fall back to CPU if CUDA SVD fails', default=True)
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
        
        # Process all tensor files
        tensor_files = list(grid_dir.glob("*.npy"))
        
        # Process tensors
        success_count = 0
        error_count = 0
        
        for tensor_file in tqdm(tensor_files, desc=f"Processing tensors in {grid_dir.name}", 
                              disable=not args.verbose, leave=False):
            tensor_dir = grid_output_dir / tensor_file.stem
            tensor_dir.mkdir(exist_ok=True)
            
            try:
                # Process BitArray tensors differently if they come from Julia
                logging.info(f"Loading tensor from {tensor_file.name}")
                np_tensor = load_tensor_safely(tensor_file)
                
                # Print info about tensor data type and range
                logging.info(f"Tensor dtype: {np_tensor.dtype}, shape: {np_tensor.shape}")
                logging.info(f"Value range: min={np_tensor.min()}, max={np_tensor.max()}")
                
                # Convert to torch tensor
                current_device = device
                if device.type == 'cuda':
                    tensor = torch.from_numpy(np_tensor).to(device=device, dtype=torch.float32)
                else:
                    tensor = torch.from_numpy(np_tensor)
                
                # Try decomposition with specified device
                try:
                    # Find optimal rank if requested
                    if args.find_rank:
                        optimal_rank, misclassification, f1_error, mse, rel_error = find_optimal_rank(
                            tensor, 
                            max_rank=args.rank,
                            device=current_device,
                            verbose=args.verbose
                        )
                        logging.info(f"Found optimal rank: {optimal_rank}")
                        rank = optimal_rank
                    else:
                        rank = args.rank
                    
                    # Decompose and save
                    result = decompose_tensor(tensor, tensor_dir, tensor_file.name, rank=rank, verbose=args.verbose, device=current_device)
                    
                except Exception as e:
                    # If CUDA failed and fallback is enabled, try with CPU
                    if "cusolver error" in str(e) and args.fall_back_to_cpu and current_device.type == 'cuda':
                        logging.warning(f"⚠️ CUDA SVD failed for {tensor_file.name}, falling back to CPU")
                        current_device = torch.device('cpu')
                        
                        # Move tensor to CPU
                        tensor = tensor.cpu()
                        
                        # Retry with CPU
                        if args.find_rank:
                            optimal_rank, misclassification, f1_error, mse, rel_error = find_optimal_rank(
                                tensor, 
                                max_rank=args.rank,
                                device=current_device,
                                verbose=args.verbose
                            )
                            logging.info(f"Found optimal rank on CPU: {optimal_rank}")
                            rank = optimal_rank
                        else:
                            rank = args.rank
                        
                        # Decompose and save with CPU
                        result = decompose_tensor(tensor, tensor_dir, tensor_file.name, rank=rank, verbose=args.verbose, device=current_device)
                    else:
                        # Re-raise if not a CUDA error or if fallback is disabled
                        raise
                
                if "error" not in result:
                    success_count += 1
                    logging.info(f"Successfully processed {tensor_file.name}")
                else:
                    error_count += 1
                    logging.error(f"Failed to decompose {tensor_file.name}: {result['error']}")
                    if not args.skip_errors:
                        raise RuntimeError(f"Failed to decompose {tensor_file.name}")
                
                # Clear the tensor from memory
                del tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                error_count += 1
                logging.error(f"❌ Error processing {tensor_file.name}: {str(e)}")
                if args.skip_errors:
                    continue
                else:
                    raise
            
        logging.info(f"Completed processing {grid_dir.name} - Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main() 