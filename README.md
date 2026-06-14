---
title: Serengeti Wildlife Classifier API
emoji: 🦓
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Serengeti Wildlife Classifier — API

A FastAPI backend serving a ResNet18 model fine-tuned to classify camera trap
images from the Serengeti into 11 classes (10 species + empty frames).

Part of a larger project covering the full pipeline — data exploration,
preprocessing, training, and experimentation. Full write-up and training code:
[GitHub repo link].

## Model

- **Architecture:** ResNet18 (ImageNet pretrained, fully fine-tuned)
- **Test accuracy:** 84.0% across 11 classes
- **Classes:** blank, buffalo, elephant, gazelleThomsons, giraffe, guineaFowl,
  hippopotamus, hyenaSpotted, lionFemale, wildebeest, zebra

## Endpoints

### `GET /`
Health check. Returns model status and the list of classes.

### `POST /predict`
Upload an image, get back the top-3 predicted species with confidence scores.

**Request:** multipart form-data, field name `file`, image file (jpg/png)

**Response:**
```json
{
  "predictions": [
    {"species": "zebra", "confidence": 91.42},
    {"species": "wildebeest", "confidence": 5.13},
    {"species": "buffalo", "confidence": 1.87}
  ]
}
```

Try it interactively at `/docs`.

## A note on what this model is good at

This model was trained exclusively on **camera trap imagery** — fixed cameras,
motion-triggered, animals at a distance, natural lighting and framing.

It performs well on images with that style. It performs noticeably worse on
close-up, posed wildlife photography (different framing, lighting, and
backgrounds than camera trap data) — this is a known limitation and is discussed
in detail in the project write-up.

For best results, try images that look like they could come from a camera trap:
animals at a moderate distance, natural outdoor settings, not posed close-ups.

## Tech

PyTorch · torchvision · OpenCV · FastAPI · Docker
