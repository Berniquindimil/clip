import os
import torch
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, BlipForImageTextRetrieval
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

dataset_dir = "dataset"
csv_file = "captions.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_file)

model_name = "Salesforce/blip-itm-base-coco"
model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
processor = BlipProcessor.from_pretrained(model_name)

image_embs, text_embs = [], []

for _, row in df.iterrows():
    img_path = os.path.join(dataset_dir, row["category"], row["filename"])
    image = Image.open(img_path).convert("RGB")
    caption = row["caption"]

    inputs = processor(images=image, text=caption, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # Extraer features visuales del encoder
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        image_feat = vision_outputs.pooler_output  # [CLS] del encoder visual
        image_emb = model.vision_proj(image_feat)  # proyectado al espacio multimodal

        # Extraer features textuales del encoder
        text_outputs = model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]  # [CLS]
        text_emb = model.text_proj(text_feat)

        # Normalizar
        img_emb = F.normalize(image_emb, dim=-1)
        txt_emb = F.normalize(text_emb, dim=-1)

    image_embs.append(img_emb)
    text_embs.append(txt_emb)

# Similaridad global
image_embs = torch.cat(image_embs, dim=0)
text_embs = torch.cat(text_embs, dim=0)
similarities = image_embs @ text_embs.T

top1_correct = sum(similarities[i].argmax().item() == i for i in range(similarities.size(0)))
top1_acc = top1_correct / similarities.size(0)
print(f"Top-1 accuracy (BLIP ITM embeddings): {top1_acc*100:.2f}%")

# Visualización
sim_df = pd.DataFrame(similarities.cpu().numpy(), index=df["filename"], columns=df["filename"])
sns.heatmap(sim_df, cmap="coolwarm")
plt.title("BLIP ITM Similarity")
plt.show()
