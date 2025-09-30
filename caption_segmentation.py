# caption_segmentation.py
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"

# Image Captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Segmentation model
seg_model = maskrcnn_resnet50_fpn(weights="DEFAULT").to(device)
seg_model.eval()

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def segment_image(image):
    img_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        pred = seg_model([img_tensor])[0]
    masks = pred['masks'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Threshold masks
    threshold = 0.5
    combined_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    
    for i in range(len(masks)):
        if scores[i] > 0.5:
            mask = masks[i, 0] > threshold
            color = np.random.randint(0, 255, (3,))
            combined_mask[mask] = color
    
    segmented_image = Image.fromarray(combined_mask)
    return segmented_image
