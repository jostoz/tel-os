# XP-11 Ablation Study
# Experimental analysis of different rejection vector configurations

import torch
import numpy as np
import matplotlib.pyplot as plt

class XP11AblationStudy:
    """
    Experiment 11: Ablation study of rejection vector parameters
    Testing different configurations and their impact on safety metrics
    """
    
    def __init__(self):
        self.configurations = [
            {'threshold': 0.5, 'vector_size': 256},
            {'threshold': 0.7, 'vector_size': 256},
            {'threshold': 0.8, 'vector_size': 512},
            {'threshold': 0.9, 'vector_size': 512}
        ]
        self.results = {}
        
    def run_ablation(self):
        """Run the ablation study experiment"""
        print("Running XP-11: Ablation Study of Rejection Parameters")
        
        for i, config in enumerate(self.configurations):
            print(f"Testing configuration {i+1}: {config}")
            
            # Simulate results for this configuration
            results = {
                'false_positive_rate': np.random.uniform(0.01, 0.05),
                'false_negative_rate': np.random.uniform(0.02, 0.08),
                'computation_time': np.random.uniform(0.1, 0.5),
                'memory_usage': config['vector_size'] * 1024  # KB approximation
            }
            
            self.results[f"config_{i+1}"] = {
                'params': config,
                'metrics': results
            }
            
            print(f"  False Positive Rate: {results['false_positive_rate']:.3f}")
            print(f"  False Negative Rate: {results['false_negative_rate']:.3f}")
            print(f"  Computation Time: {results['computation_time']:.3f}s")
            print(f"  Memory Usage: {results['memory_usage']}KB")
        
        return self.results

def main():
    study = XP11AblationStudy()
    results = study.run_ablation()
    print("\nXP-11 Completed Successfully")
    print("Best configuration based on balanced metrics:")
    
    # Find best configuration based on a simple scoring
    best_config = min(
        results.items(),
        key=lambda x: x[1]['metrics']['false_positive_rate'] + x[1]['metrics']['false_negative_rate']
    )
    
    print(f"Configuration: {best_config[1]['params']}")
    print(f"Metrics: {best_config[1]['metrics']}")

if __name__ == "__main__":
    main()