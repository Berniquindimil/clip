# vlm_flava_dataset.py
import os
import torch
from PIL import Image
import pandas as pd
from transformers import FlavaProcessor, FlavaModel
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Configuración
dataset_dir = "dataset"
csv_file = "captions.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar dataset
df = pd.read_csv(csv_file)
captions = df["caption"].tolist()
img_paths = [os.path.join(dataset_dir, row["category"], row["filename"]) for _, row in df.iterrows()]
images = [Image.open(p).convert("RGB") for p in img_paths]

# Procesador y modelo
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
model.eval()

# Preprocesar batch completo
inputs = processor(
    text=captions,
    images=images,
    return_tensors="pt",
    padding="max_length",
    max_length=77  # longitud máxima para todos los textos
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Obtener embeddings
with torch.no_grad():
    outputs = model(**inputs)
    # Colapsar la dimensión de tokens/patches para obtener vectores de tamaño fijo
    image_embeddings = F.normalize(outputs.image_embeddings.mean(dim=1), dim=-1)       # [batch, hidden]
    text_embeddings = F.normalize(outputs.text_embeddings.mean(dim=1), dim=-1)         # [batch, hidden]
    multimodal_embeddings = F.normalize(outputs.multimodal_embeddings.mean(dim=1), dim=-1)  # [batch, hidden]

# Mostrar shapes
print("Image embeddings shape:", image_embeddings.shape)
print("Text embeddings shape:", text_embeddings.shape)
print("Multimodal embeddings shape:", multimodal_embeddings.shape)

# Guardar embeddings en disco
torch.save(image_embeddings, "image_embeddings.pt")
torch.save(text_embeddings, "text_embeddings.pt")
torch.save(multimodal_embeddings, "multimodal_embeddings.pt")

# Visualzación del resultado
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
image_2d = tsne.fit_transform(image_embeddings.numpy())

plt.figure(figsize=(6,6))
plt.scatter(image_2d[:,0], image_2d[:,1])
for i, fname in enumerate(df["filename"]):
    plt.text(image_2d[i,0]+0.1, image_2d[i,1]+0.1, fname, fontsize=8)
plt.title("Proyección 2D de embeddings de imagen")
plt.show()
