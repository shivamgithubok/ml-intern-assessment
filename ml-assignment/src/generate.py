import os
import urllib.request
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Remove 'src.' because we are already in that folder
from ngram_model import TrigramModel
LOCAL_DATA_PATH = "data/example_corpus.txt"
ONLINE_FILENAME = "alice_in_wonderland.txt"
ONLINE_URL = "https://www.gutenberg.org/files/11/11-0.txt"

def get_text_data():
    """
    Prioritizes the local example corpus. 
    If it's too small or missing, downloads Alice in Wonderland.
    """
    if os.path.exists(LOCAL_DATA_PATH):
        print(f"Found local data at {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        
        # If the file is big enough, return it. 
        # If it's tiny (like just a test file), we might want the book instead.
        if len(text) > 1000: 
            return text
        else:
            print("Local file is very small. Downloading a real book for better results...")

    # Option B: Download Book (Satisfies "Extracting Data" requirement)
    if not os.path.exists(ONLINE_FILENAME):
        print(f"Downloading data from {ONLINE_URL}...")
        try:
            with urllib.request.urlopen(ONLINE_URL) as response:
                text = response.read().decode('utf-8')
            with open(ONLINE_FILENAME, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error downloading: {e}")
            return ""
    
    print(f"Using downloaded book: {ONLINE_FILENAME}")
    with open(ONLINE_FILENAME, "r", encoding="utf-8") as f:
        return f.read()

def main():
    print("--- STARTING PIPELINE ---")

    # 1. Load Data
    text = get_text_data()
    
    if not text:
        print("No data found!")
        return

    # 2. Train Model
    print(f"Training on {len(text)} characters...")
    model = TrigramModel()
    model.fit(text)

    # 3. Generate Text
    print("\n--- Generated Text Results ---")
    for i in range(1, 6):
        generated_text = model.generate()
        print(f"Generated {i}: {generated_text}")
    print("------------------------------")

if __name__ == "__main__":
    main()

# from ngram_model import TrigramModel

# def main():
#     # Create a new TrigramModel
#     model = TrigramModel()

#     # Train the model on the example corpus
#     with open("data/example_corpus.txt", "r") as f:
#         text = f.read()
#     model.fit(text)

#     # Generate new text
#     generated_text = model.generate()
#     print("Generated Text:")
#     print(generated_text)

# if __name__ == "__main__":
#     main()
