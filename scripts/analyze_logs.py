import re
from pathlib import Path
import argparse
import math

def analyze_log_file(log_path):
    """
    Find tensors with non-zero F1 error or any NaN metrics.
    """
    bad_decompositions = []
    current_tensor = None
    metrics = {}
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract tensor name - look for the specific log format
            if 'INFO - Processing tensor:' in line:
                current_tensor = line.split('Processing tensor:')[1].strip()
                metrics = {}
                continue
                
            # Extract metrics
            if current_tensor:
                if 'INFO - F1 error:' in line:
                    # Extract just the number part after "F1 error:"
                    value_str = line.split('F1 error:')[1].strip()
                    if value_str == '0.0000%':
                        metrics['f1_error'] = 0.0
                    else:
                        try:
                            metrics['f1_error'] = float(value_str.replace('%', ''))
                        except ValueError:
                            metrics['f1_error'] = 0.0
                            
                elif any(metric in line for metric in ['INFO - MSE:', 'INFO - Relative error:', 'INFO - Misclassification rate:']):
                    if 'nan' in line.lower():
                        metrics['has_nan'] = True
                
                # Check for successful processing
                if 'INFO - Successfully processed' in line and metrics:
                    if metrics.get('f1_error', 0.0) > 0.0 or metrics.get('has_nan', False):
                        bad_decompositions.append({
                            'tensor': current_tensor,
                            'f1_error': metrics.get('f1_error', 0.0),
                            'has_nan': metrics.get('has_nan', False)
                        })
                    current_tensor = None
                    metrics = {}

    return bad_decompositions

def main():
    parser = argparse.ArgumentParser(description='Find tensors with non-zero F1 error or NaN metrics')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    
    args = parser.parse_args()
    
    bad_decomps = analyze_log_file(args.log_file)
    
    if not bad_decomps:
        print("No bad decompositions found!")
        return
        
    print("\nBad decompositions found:")
    print("-" * 80)
    for entry in bad_decomps:
        print(f"\nTensor: {entry['tensor']}")
        print(f"  F1 error: {entry['f1_error']:.4f}%")
        if entry['has_nan']:
            print("  Has NaN values in metrics")

if __name__ == '__main__':
    main()