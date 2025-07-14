import torchxrayvision as xrv
import torch
from PIL import Image
import numpy as np

def analyze_xray(image_path):
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    img = Image.open(image_path).convert("L").resize((224, 224))
    img = np.asarray(img) / 255
    img = img[None, :, :]
    img = np.repeat(img, 3, axis=0)
    img = torch.tensor(img).unsqueeze(0).float()
    model.eval()
    with torch.no_grad():
        output = model(img)
    return dict(zip(model.pathologies, output[0].detach().numpy()))
