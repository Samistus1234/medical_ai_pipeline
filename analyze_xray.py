import torch
from PIL import Image
import numpy as np
import torchxrayvision.models as xrv

def analyze_xray(image_path):
    # Load pretrained model that expects grayscale input
    model = xrv.DenseNet(weights="densenet121-res224-chex")

    # Load the uploaded image and convert it to grayscale
    img = Image.open(image_path).convert("L").resize((224, 224))

    # Normalize and shape the image
    img = np.asarray(img) / 255.0                     # shape: [224, 224]
    img = img[None, :, :]                             # shape: [1, 224, 224]
    img = torch.tensor(img).unsqueeze(0).float()      # shape: [1, 1, 224, 224]

    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(img)

    # Return predictions with disease labels
    return dict(zip(model.pathologies, output[0].detach().numpy()))
