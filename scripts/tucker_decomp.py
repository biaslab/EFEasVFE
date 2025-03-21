import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
import os
from pathlib import Path
import scipy

def calculate_parameter_count(shape, ranks):
    """Calculate the number of parameters in the Tucker decomposition."""
    # Core tensor parameters
    core_params = np.prod(ranks)
    # Factor matrix parameters
    factor_params = sum(shape[i] * ranks[i] for i in range(len(shape)))
    return core_params + factor_params

def calculate_reconstruction_error(tensor, core, factors):
    """Calculate reconstruction error and relative error."""
    reconstructed = tl.tucker_to_tensor((core, factors))
    mse = np.mean((tensor - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    relative_error = rmse / np.sqrt(np.mean(tensor ** 2))
    return rmse, relative_error

def find_optimal_ranks(tensor, max_rank=12):
    """Find ranks that give the best reconstruction quality."""
    shape = tensor.shape
    best_ranks = None
    best_error = float('inf')
    
    # Start with minimum ranks
    current_ranks = [1] * len(shape)  # Start with rank 2 for better initial approximation
    
    while True:
        try:
            core, factors = non_negative_tucker(tensor, rank=current_ranks)
            rmse, relative_error = calculate_reconstruction_error(tensor, core, factors)
            
            # Update best if this is better
            if relative_error < best_error:
                best_ranks = current_ranks.copy()
                best_error = relative_error
            
            # Try increasing each rank
            improved = False
            for i in range(len(shape)):
                if current_ranks[i] >= min(max_rank, shape[i]):
                    continue
                    
                test_ranks = current_ranks.copy()
                test_ranks[i] += 1
                
                core, factors = non_negative_tucker(tensor, rank=test_ranks)
                rmse, relative_error = calculate_reconstruction_error(tensor, core, factors)
                
                if relative_error < best_error:
                    current_ranks = test_ranks
                    improved = True
                    break
            
            if not improved:
                break
                
        except Exception as e:
            print(f"Warning: Error during decomposition with ranks {current_ranks}: {str(e)}")
            # If we have a valid best_ranks, use that
            if best_ranks is not None:
                return best_ranks, best_error
            # Otherwise, try with slightly larger ranks
            current_ranks = [min(r + 1, max_rank) for r in current_ranks]
            if all(r >= max_rank for r in current_ranks):
                # If we've hit max_rank everywhere, return the last valid ranks
                return current_ranks, float('inf')
    
    # Ensure we always return valid ranks
    if best_ranks is None:
        best_ranks = current_ranks
    
    return best_ranks, best_error

def decompose_tensor(tensor, output_dir, tensor_name):
    """Decompose a single tensor and save its components."""
    print(f"Finding optimal ranks for {tensor_name}...")
    ranks, relative_error = find_optimal_ranks(tensor)
    
    # Perform final decomposition with optimal ranks
    core, factors = non_negative_tucker(tensor, rank=ranks)
    rmse, final_error = calculate_reconstruction_error(tensor, core, factors)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Optimal ranks: {ranks}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Relative error: {final_error:.4%}")
    
    # Save results
    np.save(output_dir / "core.npy", core)
    for i, factor in enumerate(factors):
        np.save(output_dir / f"factor_{i}.npy", factor)
    
    # Save metadata
    metadata = {
        'shape': tensor.shape,
        'ranks': ranks,
        'rmse': float(rmse),
        'relative_error': float(final_error)
    }
    np.save(output_dir / "metadata.npy", metadata)
    
    print(f"Saved decomposition for {tensor_name}")

def main():
    # Set the backend to scipy explicitly
    tl.set_backend('numpy')
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/tucker_decomposed_tensors")
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