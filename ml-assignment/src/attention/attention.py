import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes the Scaled Dot-Product Attention.
    
    Formula: Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    
    Args:
        Q (np.array): Queries matrix of shape (seq_len, d_k)
        K (np.array): Keys matrix of shape (seq_len, d_k)
        V (np.array): Values matrix of shape (seq_len, d_v)
        mask (np.array, optional): Mask to prevent attention to certain positions.
                                   Shape should be broadcastable to (seq_len, seq_len).
                                   Usually 1 for "keep" and 0 for "mask" (or -inf).
    
    Returns:
        output (np.array): The weighted sum of values. Shape (seq_len, d_v)
        attention_weights (np.array): The normalized attention scores. Shape (seq_len, seq_len)
    """
    
    matmul_qk = np.matmul(Q, K.T)
    d_k = Q.shape[-1]
    scaled_scores = matmul_qk / np.sqrt(d_k)
    
    # 3. Apply Mask (Optional)
    if mask is not None:
        # Where the mask is 1 (or True), we keep the value.
        # Where the mask is 0 (or False), we set it to a huge negative number (-1e9).
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)
    
    # 4. Apply Softmax
    # We want the scores to turn into probabilities (sum to 1).
    # We subtract the max for numerical stability (prevents overflow).
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights