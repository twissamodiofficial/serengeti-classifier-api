import torch
import torch.nn as nn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from gradcam import GradCAM, overlay_heatmap
from utils.preprocessing_utils import preprocess_image
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('model/best_model.pth', map_location=device)
classes = checkpoint['classes']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
grad_cam = GradCAM(model, model.layer4[-1])

# Same normalization as training 
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


@app.get("/")
def root():
    return {"status": "ok", "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not decode image"}

    processed = preprocess_image(img)

    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    tensor = normalize(processed_rgb).unsqueeze(0).to(device)
    '''
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    '''

    cam, class_idx, outputs = grad_cam.generate(tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    overlay = overlay_heatmap(processed, cam)
    success, encoded_bytes = cv2.imencode('.jpg', overlay)
    base64_bytes = None
    if success:
        base64_bytes = base64.b64encode(encoded_bytes).decode('utf-8')

    results = [
        {"species": classes[idx], "confidence": round(prob.item() * 100, 2)}
        for prob, idx in zip(top3_probs, top3_idx)
    ]

    return {
        "predictions": results,
        "overlay": base64_bytes,
        "predicted_class": classes[class_idx]
    }