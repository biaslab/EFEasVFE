import numpy as np
import tensorly as tl
import tensorly.contrib.sparse as stl
from tensorly.contrib.sparse.decomposition import parafac as sparse_parafac
import os
from pathlib import Path
import torch
import sparse
import argparse
from tqdm import tqdm
import logging
from typing import Optional, Tuple, List, Dict, Any
import datetime

# Reuse your existing setup_logging function
def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration to write to a file."""
    level = logging.DEBUG if verbose else logging.INFO
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sparse_parafac_decomp_{timestamp}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to file: {log_file}")

def load_and_convert_to_sparse(file_path: Path) -> sparse.COO:
    """Load tensor data and convert to sparse format."""
    try:
        # Load the raw data
        data = np.load(file_path)
        
        # Print detailed info about the loaded tensor
        logging.info(f"Loaded tensor from {file_path.name}:")
        logging.info(f"  - Shape: {data.shape}")
        logging.info(f"  - Dtype: {data.dtype}")
        
        # Convert to float32
        data = data.astype(np.float32)
        
        # Convert to sparse COO format
        sparse_tensor = sparse.COO.from_numpy(data)
        
        # Print sparsity information
        nnz = sparse_tensor.nnz
        total_elements = np.prod(sparse_tensor.shape)
        sparsity = 1 - (nnz / total_elements)
        
        logging.info(f"Sparse tensor statistics:")
        logging.info(f"  - Non-zero elements: {nnz}")
        logging.info(f"  - Total elements: {total_elements}")
        logging.info(f"  - Sparsity: {sparsity:.4%}")
        
        return sparse_tensor
        
    except Exception as e:
        logging.error(f"Error loading tensor from {file_path}: {str(e)}")
        raise

def calculate_sparse_reconstruction_error(
    original: sparse.COO,
    weights: np.ndarray,
    factors: List[sparse.COO]
) -> Tuple[float, float, float, float]:
    """Calculate error metrics for sparse tensor reconstruction."""
    # Convert to TensorLy sparse tensor format
    original_tl = stl.tensor(original)
    
    # Reconstruct using CP format
    reconstructed = stl.cp_to_tensor((weights, factors))
    
    # Calculate binary classification metrics
    binary_reconstruction = sparse.COO.from_numpy((reconstructed.todense() > 0.5).astype(float))
    misclassification = float(np.mean((binary_reconstruction != original).todense()))
    
    # Calculate F1 score components
    true_positives = float(((binary_reconstruction == 1) & (original == 1)).sum())
    predicted_positives = float((binary_reconstruction == 1).sum())
    actual_positives = float((original == 1).sum())
    
    precision = true_positives / (predicted_positives + 1e-10)
    recall = true_positives / (actual_positives + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Calculate reconstruction errors
    mse = float(((reconstructed - original) ** 2).mean())
    rel_error = float(np.linalg.norm((reconstructed - original).todense()) / np.linalg.norm(original.todense()))
    
    return misclassification, 1 - f1_score, mse, rel_error

def decompose_sparse_tensor(
    tensor: sparse.COO,
    output_dir: Path,
    tensor_name: str,
    rank: int = 1000,
    verbose: bool = False
) -> Dict[str, Any]:
    """Decompose a sparse tensor and save its components."""
    logging.info(f"Processing sparse tensor: {tensor_name}")
    
    try:
        # Convert to TensorLy sparse tensor format
        tensor_tl = stl.tensor(tensor)
        
        # Perform sparse PARAFAC decomposition
        weights, factors = sparse_parafac(
            tensor_tl,
            rank=rank,
            init='random',  # SVD initialization might be too memory-intensive
            tol=1e-6,
            n_iter_max=100,
            verbose=verbose
        )
        
        # Calculate reconstruction metrics
        misclassification, f1_error, mse, rel_error = calculate_sparse_reconstruction_error(
            tensor, weights, factors
        )
        
        logging.info(f"Decomposition results:")
        logging.info(f"  - Misclassification rate: {misclassification:.4%}")
        logging.info(f"  - F1 error: {f1_error:.4%}")
        logging.info(f"  - MSE: {mse:.4e}")
        logging.info(f"  - Relative error: {rel_error:.4e}")
        
        # Save results
        np.save(output_dir / "weights.npy", weights)
        for i, factor in enumerate(factors):
            np.save(output_dir / f"factor_{i}.npy", factor.todense())
        
        # Save metadata
        metadata = {
            'shape': tensor.shape,
            'rank': rank,
            'nnz': tensor.nnz,
            'sparsity': 1 - (tensor.nnz / np.prod(tensor.shape)),
            'misclassification': float(misclassification),
            'f1_error': float(f1_error),
            'mse': float(mse),
            'relative_error': float(rel_error)
        }
        np.save(output_dir / "metadata.npy", metadata)
        
        logging.info(f"Saved sparse decomposition for {tensor_name}")
        return metadata
        
    except Exception as e:
        logging.error(f"Error during sparse decomposition of {tensor_name}: {str(e)}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Perform sparse PARAFAC decomposition on tensors')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--rank', type=int, default=1000, help='Rank for PARAFAC decomposition')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation even if decomposition exists')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for failed decompositions')
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path("data/parafac_decomposed_sparse_tensors")
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
        
        for tensor_file in tqdm(tensor_files, desc=f"Processing tensors in {grid_dir.name}", 
                              disable=not args.verbose):
            tensor_dir = grid_output_dir / tensor_file.stem
            tensor_dir.mkdir(exist_ok=True)
            
            # Skip if already processed and not forcing recompute
            if not args.force_recompute and (tensor_dir / "metadata.npy").exists():
                logging.info(f"Skipping {tensor_file.name} - already processed")
                continue
            
            retries = 0
            while retries < args.max_retries:
                try:
                    # Load and convert to sparse format
                    sparse_tensor = load_and_convert_to_sparse(tensor_file)
                    
                    # Decompose and save
                    result = decompose_sparse_tensor(
                        sparse_tensor,
                        tensor_dir,
                        tensor_file.name,
                        rank=args.rank,
                        verbose=args.verbose
                    )
                    
                    if "error" not in result:
                        logging.info(f"Successfully processed {tensor_file.name}")
                        break
                    else:
                        retries += 1
                        logging.error(f"Attempt {retries}/{args.max_retries} failed: {result['error']}")
                        
                except Exception as e:
                    retries += 1
                    logging.error(f"Attempt {retries}/{args.max_retries} failed with error: {str(e)}")
                    if retries < args.max_retries:
                        logging.info(f"Retrying...")
                    else:
                        logging.error(f"Failed to process {tensor_file.name} after {args.max_retries} attempts")

if __name__ == "__main__":
    main()