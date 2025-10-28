import os
import torch
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
dataset_dir = "dataset"  # Carpeta con subcarpetas de imágenes
csv_file = "captions.csv"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ----------------------------
# CARGAR CSV
# ----------------------------
df = pd.read_csv(csv_file)

# ----------------------------
# CARGAR MODELO CLIP
# ----------------------------
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# ----------------------------
# EMBEDDINGS DE IMÁGENES Y TEXTOS
# ----------------------------
image_embeddings = []
text_embeddings = []

for idx, row in df.iterrows():
    # Imagen
    category = row['category']
    filename = row['filename']
    img_path = os.path.join(dataset_dir, category, filename)
    image = Image.open(img_path).convert("RGB")

    # Procesar imagen y caption
    inputs = processor(text=row['caption'], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # image_embedding: [1, 512], text_embedding: [1, 512]
        image_emb = F.normalize(outputs.image_embeds, p=2, dim=1)
        text_emb = F.normalize(outputs.text_embeds, p=2, dim=1)

    image_embeddings.append(image_emb)
    text_embeddings.append(text_emb)

# Convertir a tensor
image_embeddings = torch.cat(image_embeddings, dim=0)  # (num_images, 512)
text_embeddings = torch.cat(text_embeddings, dim=0)    # (num_images, 512)

# ----------------------------
# SIMILITUD COSENO
# ----------------------------
similarities = image_embeddings @ text_embeddings.T  # (num_images, num_texts)

# ----------------------------
# TOP-1 ACCURACY
# ----------------------------
top1_correct = 0
for i in range(similarities.size(0)):
    top_idx = similarities[i].argmax().item()
    if top_idx == i:
        top1_correct += 1

top1_acc = top1_correct / similarities.size(0)
print(f"Top-1 accuracy (image matches its caption with CLIP): {top1_acc*100:.2f}%")

# ----------------------------
# MATRIZ DE SIMILITUD
# ----------------------------
import pandas as pd
sim_matrix = similarities.cpu().numpy()
sim_df = pd.DataFrame(sim_matrix, index=df['filename'], columns=df['filename'])
print("\nCosine similarity matrix (images x captions) with CLIP:")
print(sim_df.round(3))
