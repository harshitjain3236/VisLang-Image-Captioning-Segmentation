import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Mask R-CNN for segmentation
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Transform for image
transform = T.Compose([T.ToTensor()])

def segment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])

    masks = outputs[0]['masks'] > 0.5
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    img_np = np.array(image)
    for i in range(len(masks)):
        if scores[i] > 0.5:
            mask = masks[i][0].cpu().numpy()
            color = np.random.randint(0, 255, (3,))
            img_np[mask] = img_np[mask] * 0.5 + color * 0.5

    segmented_image = Image.fromarray(img_np.astype(np.uint8))
    return segmented_image
