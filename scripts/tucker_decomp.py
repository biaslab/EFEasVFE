import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
import os
from pathlib import Path
import scipy

def decompose_tensor(tensor, output_dir, tensor_name):
    """Decompose a single tensor and save its components."""
    # Get tensor shape to determine appropriate ranks
    shape = tensor.shape
    # Use min(6, shape[i]) for each dimension to ensure rank doesn't exceed dimension size
    ranks = [min(6, shape[i]) for i in range(len(shape))]
    print(f"Tensor shape: {shape}, Using ranks: {ranks}")
    
    # Perform non-negative Tucker decomposition
    core, factors = non_negative_tucker(tensor, rank=ranks)
    
    # Save results
    # Save core tensor
    np.save(output_dir / "core.npy", core)
    # Save each factor matrix separately
    for i, factor in enumerate(factors):
        np.save(output_dir / f"factor_{i}.npy", factor)
    
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