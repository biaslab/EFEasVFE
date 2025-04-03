import numpy as np
import torch
import tensorly as tl
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, List
import datetime

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a timestamp for the log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"decomposition_test_{timestamp}.log"
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to file: {log_file}")

def check_decomposition(decomp_dir: Path) -> Dict[str, Any]:
    """Check a single tensor decomposition."""
    try:
        # Load metadata
        metadata = np.load(decomp_dir / "metadata.npy", allow_pickle=True).item()
        
        # Check if original metrics indicate problems
        if metadata.get('f1_error', 0.0) > 0.0 or np.isnan(metadata.get('mse', 0.0)):
            return {
                "status": "warning",
                "warning": "Original decomposition had high F1 error or NaN metrics",
                "metrics": {
                    "f1_error": metadata.get('f1_error', None),
                    "mse": metadata.get('mse', None),
                    "misclassification": metadata.get('misclassification', None),
                    "relative_error": metadata.get('relative_error', None)
                }
            }
        
        return {
            "status": "ok",
            "shape": metadata['shape'],
            "rank": metadata['rank'],
            "original_metrics": {
                "misclassification": metadata.get('misclassification', None),
                "f1_error": metadata.get('f1_error', None),
                "mse": metadata.get('mse', None),
                "relative_error": metadata.get('relative_error', None)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Test tensor decompositions')
    parser.add_argument('--decomp-dir', type=str, default='data/parafac_decomposed_tensors',
                       help='Directory containing decomposed tensors')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    decomp_dir = Path(args.decomp_dir)
    
    if not decomp_dir.exists():
        logging.error(f"Directory not found: {decomp_dir}")
        return
    
    # Process each grid size directory
    problems_found = []
    warnings_found = []
    
    for grid_dir in decomp_dir.iterdir():
        if not grid_dir.is_dir():
            continue
            
        logging.info(f"\nChecking grid size directory: {grid_dir.name}")
        
        # Check each tensor decomposition
        for tensor_dir in grid_dir.iterdir():
            if not tensor_dir.is_dir():
                continue
                
            result = check_decomposition(tensor_dir)
            
            if result["status"] == "error":
                problems_found.append({
                    "tensor": tensor_dir.name,
                    "grid": grid_dir.name,
                    "error": result["error"]
                })
                logging.error(f"❌ Problem with {tensor_dir.name}: {result['error']}")
            elif result["status"] == "warning":
                warnings_found.append({
                    "tensor": tensor_dir.name,
                    "grid": grid_dir.name,
                    "metrics": result["metrics"]
                })
                logging.warning(f"⚠️ Warning for {tensor_dir.name}: High F1 error or NaN metrics")
                if args.verbose:
                    for metric, value in result["metrics"].items():
                        logging.warning(f"  {metric}: {value}")
            else:
                logging.info(f"✓ {tensor_dir.name} - Rank: {result['rank']}")
                if args.verbose:
                    logging.debug(f"  Original metrics:")
                    for metric, value in result['original_metrics'].items():
                        logging.debug(f"    {metric}: {value}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if problems_found:
        print("\n🚨 Problems found:")
        for problem in problems_found:
            print(f"- {problem['grid']}/{problem['tensor']}: {problem['error']}")
    
    if warnings_found:
        print("\n⚠️ Warnings found:")
        for warning in warnings_found:
            print(f"- {warning['grid']}/{warning['tensor']}:")
            for metric, value in warning['metrics'].items():
                print(f"  {metric}: {value}")
    
    if not problems_found and not warnings_found:
        print("\n✅ All decompositions look good!")
    
    print("\nTotal statistics:")
    print(f"- Problems: {len(problems_found)}")
    print(f"- Warnings: {len(warnings_found)}")

if __name__ == "__main__":
    main()
