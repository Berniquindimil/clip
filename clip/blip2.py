import os
import torch
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_dir = "dataset"
csv_file = "captions.csv"

df = pd.read_csv(csv_file)

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

generated_captions = []

for _, row in df.iterrows():
    img_path = os.path.join(dataset_dir, row["category"], row["filename"])
    image = Image.open(img_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=30)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

    generated_captions.append(caption)
    print(row["filename"], " → ", caption)

df["generated_caption"] = generated_captions
df.to_csv("results_with_generated_captions.csv", index=False)

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

references = df["caption"].apply(lambda x: [x]).tolist()  # formato [["ref1"], ["ref2"], ...]
predictions = df["generated_caption"].tolist()

bleu_score = bleu.compute(predictions=predictions, references=references)
meteor_score = meteor.compute(predictions=predictions, references=references)

print("BLEU:", bleu_score)
print("METEOR:", meteor_score)