from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2", filename="config.json")
print(f"Model downloaded to: {model_path}")