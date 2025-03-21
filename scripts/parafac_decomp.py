import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import os
from pathlib import Path
import scipy

def calculate_parameter_count(shape, rank):
    """Calculate the number of parameters in the PARAFAC decomposition."""
    # Each factor matrix has shape (dimension_size, rank)
    factor_params = sum(shape[i] * rank for i in range(len(shape)))
    return factor_params

def calculate_reconstruction_error(tensor, weights, factors):
    """Calculate error metrics appropriate for binary tensors."""
    reconstructed = tl.cp_to_tensor((weights, factors))
    # Threshold the reconstruction to binary
    binary_reconstruction = (reconstructed > 0.5).astype(tensor.dtype)
    # Calculate misclassification rate
    misclassification = np.mean(binary_reconstruction != tensor)
    # Calculate F1 score
    true_positives = np.sum((binary_reconstruction == 1) & (tensor == 1))
    precision = true_positives / (np.sum(binary_reconstruction == 1) + 1e-10)
    recall = true_positives / (np.sum(tensor == 1) + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return misclassification, 1 - f1_score

def find_optimal_rank(tensor, max_rank=20):
    """Find rank that gives the best reconstruction quality."""
    shape = tensor.shape
    best_rank = None
    best_misclassification = float('inf')
    best_f1_error = float('inf')
    
    # Start with minimum rank
    current_rank = 1
    
    while current_rank <= max_rank:
        try:
            # Add memory-efficient options
            weights, factors = parafac(tensor, rank=current_rank, 
                                    init='random',  # Use random initialization
                                    tol=1e-6,      # Slightly relaxed tolerance
                                    n_iter_max=1000)  # Limit iterations
            
            misclassification, f1_error = calculate_reconstruction_error(tensor, weights, factors)
            
            # Update best if this is better
            if misclassification < best_misclassification:
                best_rank = current_rank
                best_misclassification = misclassification
                best_f1_error = f1_error
                print(f"Found better rank {current_rank}: misclassification={misclassification:.4f}, F1_error={f1_error:.4f}")
            
            # If we have perfect reconstruction, stop
            if misclassification == 0 and f1_error == 0:
                print("Perfect reconstruction achieved, stopping.")
                break
            
            current_rank += 1
                
        except Exception as e:
            print(f"Warning: Error during decomposition with rank {current_rank}: {str(e)}")
            # If we have a valid best_rank, use that
            if best_rank is not None:
                return best_rank, best_misclassification, best_f1_error
            # Otherwise, try next rank
            current_rank += 1
    
    # Ensure we always return valid rank
    if best_rank is None:
        best_rank = 1  # Default to rank 1 if all attempts failed
    
    return best_rank, best_misclassification, best_f1_error

def decompose_tensor(tensor, output_dir, tensor_name):
    """Decompose a single tensor and save its components."""
    print(f"Finding optimal rank for {tensor_name}...")
    
    # Check if tensor is binary
    unique_values = np.unique(tensor)
    is_binary = len(unique_values) == 2 and set(unique_values) == {0, 1}
    if not is_binary:
        print("Warning: Tensor is not binary, results may be unreliable")
    
    # rank, misclassification, f1_error = find_optimal_rank(tensor)
    rank = 100
    
    # Perform final decomposition with optimal rank
    weights, factors = parafac(tensor, rank=rank, 
                              init='random',
                              tol=1e-6,
                              n_iter_max=100)
    final_misclassification, final_f1_error = calculate_reconstruction_error(tensor, weights, factors)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Optimal rank: {rank}")
    print(f"Misclassification rate: {final_misclassification:.4%}")
    print(f"F1 error: {final_f1_error:.4%}")
    
    # Save results
    np.save(output_dir / "weights.npy", weights)
    for i, factor in enumerate(factors):
        np.save(output_dir / f"factor_{i}.npy", factor)
    
    # Save metadata
    metadata = {
        'shape': tensor.shape,
        'rank': rank,
        'misclassification': float(final_misclassification),
        'f1_error': float(final_f1_error),
        'is_binary': is_binary
    }
    np.save(output_dir / "metadata.npy", metadata)
    
    print(f"Saved decomposition for {tensor_name}")

def main():
    # Set the backend to scipy explicitly
    tl.set_backend('numpy')
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/parafac_decomposed_tensors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the tensors from raw_tensors
    input_path = Path("data/raw_tensors")
    
    # Process each grid size directory
    for grid_dir in input_path.iterdir():
        if not grid_dir.is_dir():
            continue
            
        print(f"\nProcessing grid size directory: {grid_dir.name}")
        
        # Create corresponding output directory
        grid_output_dir = output_dir / grid_dir.name
        grid_output_dir.mkdir(exist_ok=True)
        
        # Process each tensor in the grid size directory
        for tensor_file in grid_dir.glob("*.npy"):
            print(f"Processing {tensor_file.name}...")
            
            # Create a separate directory for this tensor's decomposition
            tensor_dir = grid_output_dir / tensor_file.stem
            tensor_dir.mkdir(exist_ok=True)
            
            # Load tensor
            tensor = np.load(tensor_file)
            
            # Decompose and save
            decompose_tensor(tensor, tensor_dir, tensor_file.name)
            
        print(f"Completed processing {grid_dir.name}")

if __name__ == "__main__":
    main() 