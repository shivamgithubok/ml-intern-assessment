import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention import scaled_dot_product_attention

def run_demo():
    print("--- Scaled Dot-Product Attention Demo ---")
    
    # 1. Setup random data
    # Sequence length = 3 (e.g., "I love AI")
    np.random.seed(42) 
    
    Q = np.random.rand(3, 4)
    K = np.random.rand(3, 4)
    V = np.random.rand(3, 4)
    
    print(f"Query Shape: {Q.shape}")
    print(f"Key Shape:   {K.shape}")
    print(f"Value Shape: {V.shape}")
    
    # 2. Run Attention WITHOUT mask
    print("\nRunning Attention (No Mask)...")
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nCalculated Attention Weights (should sum to 1 per row):")
    print(np.round(weights, 2))
    
    print("\nOutput (Weighted Values):")
    print(np.round(output, 2))
    
    # 3. Run Attention WITH mask
    print("\n--------------------------------")
    print("Running Attention (WITH Mask)...")
    
    # Mask: 1 means keep, 0 means hide
    # A causal mask (look-ahead mask) looks like a lower triangular matrix
    mask = np.array([
        [1, 0, 0],  
        [1, 1, 0],  
        [1, 1, 1]   
    ])
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\nMasked Attention Weights (Notice zeros in top right):")
    print(np.round(weights_masked, 2))

if __name__ == "__main__":
    run_demo()