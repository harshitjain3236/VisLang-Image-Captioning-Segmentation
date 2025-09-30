from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

def generate_caption(image_path):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
