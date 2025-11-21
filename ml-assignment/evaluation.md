# Evaluation Report: Trigram Language Model

## Design Choices

### 1. Data Structure: Nested Dictionary
To store the N-gram counts, I chose a nested dictionary structure: `defaultdict(Counter)`.
- **Key:** A tuple representing the context `(word1, word2)`.
- **Value:** A `Counter` object mapping possible next words to their frequencies.
- **Reasoning:** This structure provides O(1) access time. When generating text, we can instantly retrieve the distribution of likely next words for any given context without iterating through the entire dataset.

### 2. Preprocessing and Tokenization
The `fit` method processes the raw text in three steps:
1.  **Lowercasing:** All text is converted to lowercase to ensure tokens like "The" and "the" are treated as the same entity, improving statistical density.
2.  **Sentence Splitting:** I split text by punctuation (`.`, `!`, `?`) rather than just newlines. This prevents the model from learning invalid transitions across sentence boundaries.
3.  **Padding:** For a Trigram model (N=3), predicting the first word of a sentence requires a context of size N-1 (2). Therefore, I padded the start of every sentence with two `<START>` tokens and appended one `<END>` token to mark termination.

### 3. Generation Strategy: Weighted Random Sampling
For the `generate` function, I did not use a "Greedy" approach (always picking the most frequent word). Instead, I used **probabilistic sampling**:
- I used `random.choices()` which selects the next word based on the probability distribution observed in the training data.
- **Reasoning:** This preserves the diversity and style of the original text. A greedy approach would result in repetitive, deterministic loops, whereas weighted sampling produces more natural and varied sentences.

### 4. Handling Unknowns and Dead Ends
If the generation process encounters a context `(w1, w2)` that was never seen during training (or reaches the end of the known path), the loop terminates immediately. This prevents the model from crashing or generating nonsense when it runs out of learned patterns.


## Task 2: Scaled Dot-Product Attention

### Implementation Details
I implemented the attention mechanism using pure Numpy, following the standard formula:
$$ Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

1.  **Scaling:** I divided the dot product $QK^T$ by $\sqrt{d_k}$. This is crucial because for large dimensions, the dot products grow large in magnitude, pushing the softmax function into regions where gradients are extremely small (vanishing gradients). Scaling keeps the variance stable.
2.  **Masking:** I implemented an optional masking step. Before the softmax, positions corresponding to `0` in the mask are set to negative infinity (`-1e9`). This ensures their probability becomes exactly zero after the softmax operation, effectively "hiding" those tokens.
3.  **Numerical Stability:** For the softmax implementation, I subtracted the maximum value from the logits before exponentiation to prevent numerical overflow errors.

### Demonstration
I created a demo script (`src/attention/demo.py`) that initializes random Q, K, and V matrices (Shape 3x4).
- **Without Mask:** The attention weights are distributed across all tokens.
- **With Mask:** Using a lower-triangular mask, the attention weights for future tokens (upper triangle) correctly dropped to 0.0, simulating a causal (decoder-style) attention block.