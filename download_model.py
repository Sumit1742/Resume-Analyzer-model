import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from sentence_transformers import SentenceTransformer

print("Downloading model... Please wait.")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("miniLM_model")

print("Model downloaded successfully!")
