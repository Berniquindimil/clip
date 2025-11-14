import os
import torch
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

dataset_dir = "dataset"
csv_file = "captions.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_file)

model_name = "facebook/flava-full"
model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

image_embeddings, text_embeddings = [], []

for _, row in df.iterrows():
    img_path = os.path.join(dataset_dir, row["category"], row["filename"])
    image = Image.open(img_path).convert("RGB")
    caption = row["caption"]

    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=caption, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask", None),
            return_dict=True
        )

        # Promediar sobre tokens/patches
        img_emb = F.normalize(outputs.image_embeddings.mean(dim=1), dim=1)
        txt_emb = F.normalize(outputs.text_embeddings.mean(dim=1), dim=1)

    image_embeddings.append(img_emb)
    text_embeddings.append(txt_emb)

image_embeddings = torch.cat(image_embeddings, dim=0)
text_embeddings = torch.cat(text_embeddings, dim=0)
similarities = image_embeddings @ text_embeddings.T

top1_correct = sum(similarities[i].argmax().item() == i for i in range(similarities.size(0)))
top1_acc = top1_correct / similarities.size(0)
print(f"Top-1 accuracy (FLAVA): {top1_acc*100:.2f}%")

sim_df = pd.DataFrame(similarities.cpu().numpy(), index=df["filename"], columns=df["filename"])
sns.heatmap(sim_df.round(2), annot=False, cmap="coolwarm")
plt.title("FLAVA Similarity")
plt.show()
