# compute_lambda.py
import os
import csv
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

DATA_DIR = "data/subsets/bdd100k_val_2000/images"
OUT_CSV  = "results/baseline_lambda.csv"
os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

hazard_prompts = [
    "a pedestrian crossing the road",
    "a cyclist in the lane",
    "a stopped vehicle blocking traffic",
    "road debris on the street",
]

safe_prompts = [
    "a clear road",
    "an empty highway",
    "a normal driving scene",
]

all_prompts = hazard_prompts + safe_prompts
text = tokenizer(all_prompts).to(device)

with torch.no_grad():
    tf = model.encode_text(text)
    tf = tf / tf.norm(dim=-1, keepdim=True)

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "lambda"])

    for fname in tqdm(files):
        img = Image.open(os.path.join(DATA_DIR, fname)).convert("RGB")
        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            imf = model.encode_image(image)
            imf = imf / imf.norm(dim=-1, keepdim=True)

        sims = (imf @ tf.T).squeeze(0).cpu()

        s_h = sims[:len(hazard_prompts)].max().item()
        s_n = sims[len(hazard_prompts):].max().item()
        lam = s_h - s_n

        writer.writerow([fname, lam])

print("Saved baseline lambda scores to", OUT_CSV)
