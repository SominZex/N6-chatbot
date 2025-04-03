from sentence_transformers import SentenceTransformer
import os
import shutil

def clear_cache(model_name):
    cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache for model: {model_name}")

def download_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model '{model_name}': {e}")

if __name__ == "__main__":
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    clear_cache(model_name)
    download_model(model_name)