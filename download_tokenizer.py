from transformers import AutoTokenizer
import os

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
SAVE_DIRECTORY = "local_tokenizer"

def download_tokenizer():
    """
    Downloads and saves the tokenizer files to a local directory.
    """
    if os.path.exists(SAVE_DIRECTORY):
        print(f"Directory '{SAVE_DIRECTORY}' already exists. Skipping download.")
        return

    print(f"Downloading tokenizer for '{MODEL_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        print(f"✅ Tokenizer successfully downloaded and saved to '{SAVE_DIRECTORY}'.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("Please check your network connection and try again.")

if __name__ == "__main__":
    download_tokenizer()
