from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    
    if not os.path.isfile(image_path):
        return f" File not found: {image_path}"

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return f" Could not open image: {e}"

    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    print("AI-Based Image Caption Generator")
    path = input("Enter the path to your image (e.g. ./cat.jpg): ").strip()
    caption = generate_caption(path)
    print("\nGenerated Caption:", caption)
