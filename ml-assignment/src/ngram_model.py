import random
from collections import Counter , defaultdict
import re
class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        self.model = defaultdict(Counter)
        
        # We keep track of known vocabulary to handle Unknown tokens later if needed
        self.vocab = set()
        pass

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # TODO: Implement the training logic.
        # This will involve:
        # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
        # 2. Tokenizing the text into words.
        # 3. Padding the text with start and end tokens.
        # 4. Counting the trigrams.
        print("Preprocessing text...")
        
        # 1. Preprocessing
        # Convert to lowercase to ensure 'The' and 'the' are treated the same
        text = text.lower()
        
        # Split text into sentences. 
        # We assume sentences end with ., !, or ?.
        sentences = re.split(r'[.!?]+', text)
        
        print(f"Training on {len(sentences)} sentences...")

        for sentence in sentences:
            # Clean the sentence: remove non-alphanumeric characters (keep spaces)
            # This regex removes anything that isn't a letter or number or whitespace
            sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
            
            # 2. Tokenization
            words = sentence.split()
            
            if not words:
                continue

            # 3. Padding
            # For a Trigram (N=3), we need N-1 (2) start tokens to predict the first word.
            tokens = ['<START>', '<START>'] + words + ['<END>']
            
            # Add to vocabulary
            self.vocab.update(tokens)

            # 4. Counting Trigrams
            # We slide a window of size 3 over the tokens
            for i in range(len(tokens) - 2):
                w1 = tokens[i]
                w2 = tokens[i+1]
                w3 = tokens[i+2]
                
                # The context is the first two words
                context = (w1, w2)
                # We record that w3 appeared after this context
                self.model[context][w3] += 1

        print("Training complete.")

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # TODO: Implement the generation logic.
        # This will involve:
        # 1. Starting with the start tokens.
        # 2. Probabilistically choosing the next word based on the current context.
        # 3. Repeating until the end token is generated or the maximum length is reached.
        # 1. Start with the start tokens
        current_context = ('<START>', '<START>')
        result = []
        
        for _ in range(max_length):
            # Check if the current context exists in our model
            if current_context not in self.model:
                break
            
            # Get the possible next words and their counts
            next_word_counts = self.model[current_context]
            
            # 2. Probabilistically choose the next word
            # extract words and their frequencies
            words = list(next_word_counts.keys())
            counts = list(next_word_counts.values())
            
            # random.choices performs weighted sampling
            # It picks a word based on how frequently it appeared in training
            next_word = random.choices(words, weights=counts, k=1)[0]
            
            # 3. Stop if end token is generated
            if next_word == '<END>':
                break
            
            # Append to results
            result.append(next_word)
            
            # Update context: slide the window forward
            # New context becomes (old_2nd_word, new_word)
            current_context = (current_context[1], next_word)
            
        return " ".join(result)