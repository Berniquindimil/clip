import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
dataset_dir = "dataset"  # Carpeta raíz con las subcarpetas de imágenes
csv_file = "captions.csv"  # Archivo con filename, category, caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CARGAR DATOS
# ----------------------------
df = pd.read_csv(csv_file)

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# MODELO DE IMAGEN: ResNet50 preentrenada
# ----------------------------
resnet = resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()
resnet.to(device)

# ----------------------------
# MODELO DE TEXTO: SentenceTransformer
# ----------------------------
text_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=device)

# ----------------------------
# OBTENER EMBEDDINGS DE IMÁGENES
# ----------------------------
image_embeddings = []
for idx, row in df.iterrows():
    category = row['category']
    filename = row['filename']
    img_path = os.path.join(dataset_dir, category, filename)
    
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = resnet(image_tensor)  # shape: (1, 2048, 1, 1)
        emb = emb.flatten(1)        # shape: (1, 2048)
        emb = F.normalize(emb, p=2, dim=1)  # normalizar
    image_embeddings.append(emb)

image_embeddings = torch.cat(image_embeddings, dim=0)  # shape: (num_images, 2048)

# ----------------------------
# OBTENER EMBEDDINGS DE TEXTOS
# ----------------------------
captions = df['caption'].tolist()
text_embeddings = text_model.encode(captions, convert_to_tensor=True, normalize_embeddings=True)  # (num_texts, dim)

# ----------------------------
# CALCULAR SIMILITUD COSENO
# ----------------------------
# Nota: ResNet y SentenceTransformer tienen dimensiones distintas, por simplicidad,
#       podemos calcular similitud coseno usando la misma librería, ajustando dimensiones si es necesario
# Para este ejemplo, reducimos imagen a 512 dimensiones con PCA simple opcional
# o usamos torch.nn.functional.cosine_similarity en pares de vectores
# Aquí vamos a hacer un truco simple: proyectar imagen embeddings a 512 con una capa lineal
proj = torch.nn.Linear(image_embeddings.size(1), text_embeddings.size(1)).to(device)
with torch.no_grad():
    image_proj = F.normalize(proj(image_embeddings), p=2, dim=1)  # shape: (num_images, 512)

# Similaridad coseno
similarities = image_proj @ text_embeddings.T  # shape: (num_images, num_texts)

# ----------------------------
# EVALUACIÓN TOP-1
# ----------------------------
top1_correct = 0
for i in range(similarities.size(0)):
    top_idx = similarities[i].argmax().item()
    if top_idx == i:
        top1_correct += 1

top1_acc = top1_correct / similarities.size(0)
print(f"Top-1 accuracy (image matches its caption): {top1_acc*100:.2f}%")

# ----------------------------
# Mostrar tabla de similitudes
# ----------------------------
sim_matrix = similarities.cpu().numpy()
sim_df = pd.DataFrame(sim_matrix, index=df['filename'], columns=df['filename'])
print("\nCosine similarity matrix (images x captions):")
print(sim_df.round(3))
