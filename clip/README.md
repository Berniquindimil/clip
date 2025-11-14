# Proyecto VLM: BLIP, CLIP y FLAVA

Este repositorio contiene scripts para trabajar con **modelos de visión-lenguaje (VLM)**: BLIP, CLIP y FLAVA, aplicados a un dataset propio de imágenes y captions.

Los modelos se ejecutan dentro de un contenedor Docker, y se proporcionan recetas de Makefile para facilitar la ejecución.

---

## 📂 Estructura del repositorio

/proyecto
├─ informes/ ← Informes y documentación
├─ clip/
│ ├─ dataset/ ← Imágenes y CSV de captions
│ ├─ blip.py ← Evaluación BLIP (ITM)
│ ├─ blip2.py ← Generación de captions con BLIP
│ ├─ clip.py ← Ejemplo CLIP
│ ├─ clip_laion.py ← CLIP Laion
│ ├─ pre-clip.py ← Preprocesamiento para CLIP
│ ├─ flava.py ← Evaluación FLAVA
│ ├─ flava2.py ← Evaluación FLAVA (multimodal)
│ ├─ Dockerfile ← Imagen Docker con dependencias
│ └─ Makefile ← Recetas para construir y ejecutar scripts


---

## ⚡ Requisitos

- Docker ≥ 20.10  
- (Opcional) GPU compatible con CUDA para acelerar inferencia  
- X11 para ejecutar GUIs dentro del contenedor (solo si es necesario)

---

## 🏗 Construcción de la imagen

Desde la carpeta `clip/`:

```bash
make build
