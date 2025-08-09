#!/usr/bin/env python3
"""
Download BrowseComp dataset from Kaggle
"""
import os
import requests
import zipfile
from pathlib import Path

def download_browsecomp():
    """Download BrowseComp dataset"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # BrowseComp dataset URL (direct download from Kaggle)
    # This is the public dataset URL
    url = "https://www.kaggle.com/api/v1/datasets/download/openai/browsecomp-a-benchmark-for-browsing-agents"
    
    print("Downloading BrowseComp dataset...")
    print("Note: This requires Kaggle authentication.")
    print("Please visit: https://www.kaggle.com/datasets/openai/browsecomp-a-benchmark-for-browsing-agents")
    print("And download the dataset manually, then place it in the data/ directory.")
    print("The file should be named 'browsecomp-a-benchmark-for-browsing-agents.zip'")
    
    # Alternative: try to download from HuggingFace datasets
    try:
        print("\nTrying to download from HuggingFace datasets...")
        from datasets import load_dataset
        
        dataset = load_dataset("openai/browsecomp")
        
        # Save as JSONL
        output_file = data_dir / "browsecomp.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset['test']:
                f.write(json.dumps(item) + '\n')
        
        print(f"Successfully downloaded BrowseComp dataset to {output_file}")
        return str(output_file)
        
    except ImportError:
        print("HuggingFace datasets not installed. Installing...")
        os.system("pip install datasets")
        
        try:
            from datasets import load_dataset
            import json
            
            dataset = load_dataset("openai/browsecomp")
            
            # Save as JSONL
            output_file = data_dir / "browsecomp.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset['test']:
                    f.write(json.dumps(item) + '\n')
            
            print(f"Successfully downloaded BrowseComp dataset to {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            print("\nPlease download manually from:")
            print("https://www.kaggle.com/datasets/openai/browsecomp-a-benchmark-for-browsing-agents")
            return None

if __name__ == "__main__":
    download_browsecomp() 