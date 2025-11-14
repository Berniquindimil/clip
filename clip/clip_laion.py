# vlm_clip_laion.py
import os
import torch
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
dataset_dir = "dataset"
csv_file = "captions.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CARGAR CSV
# ----------------------------
df = pd.read_csv(csv_file)

# ----------------------------
# CARGAR MODELO (CLIP de LAION)
# ----------------------------
model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# ----------------------------
# CALCULAR EMBEDDINGS Y SIMILITUD
# ----------------------------
image_embeddings, text_embeddings = [], []

for _, row in df.iterrows():
    category, filename = row["category"], row["filename"]
    img_path = os.path.join(dataset_dir, category, filename)
    image = Image.open(img_path).convert("RGB")
    caption = row["caption"]

    inputs = processor(text=caption, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        img_emb = F.normalize(outputs.image_embeds, p=2, dim=1)
        txt_emb = F.normalize(outputs.text_embeds, p=2, dim=1)

    image_embeddings.append(img_emb)
    text_embeddings.append(txt_emb)

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)
similarities = image_embeddings @ text_embeddings.T

# ----------------------------
# TOP-1 ACCURACY
# ----------------------------
top1_correct = sum(similarities[i].argmax().item() == i for i in range(similarities.size(0)))
top1_acc = top1_correct / similarities.size(0)
print(f"Top-1 accuracy (CLIP LAION): {top1_acc*100:.2f}%")

# ----------------------------
# MATRIZ DE SIMILITUD
# ----------------------------
sim_df = pd.DataFrame(similarities.cpu().numpy(), index=df["filename"], columns=df["filename"])
sns.heatmap(sim_df.round(2), annot=False, cmap="coolwarm")
plt.title("CLIP (LAION) Similarity")
plt.show()
