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
images from the Serengeti into 11 classes (10 species + empty frames), with
GradCAM explainability built in.

Part of a larger project covering the full pipeline — data exploration,
preprocessing, training, and experimentation:
[training repo](https://github.com/twissamodi/serengeti) ·
[frontend repo](https://github.com/twissamodi/serengeti-frontend) ·
[live demo](https://serengeti-frontend.vercel.app)

## Model

- **Architecture:** ResNet18 (ImageNet pretrained, fully fine-tuned)
- **Test accuracy:** 84.0% across 11 classes
- **Classes:** blank, buffalo, elephant, gazelleThomsons, giraffe, guineaFowl,
  hippopotamus, hyenaSpotted, lionFemale, wildebeest, zebra

## Endpoints

### `GET /`
Health check. Returns model status and the list of classes.

### `POST /predict`
Upload an image, get back the top-3 predicted species with confidence
scores, plus a GradCAM attention-map overlay showing which regions of the
image drove the prediction.

**Request:** multipart form-data, field name `file`, image file (jpg/png)

**Response:**
```json
{
  "predictions": [
    {"species": "hippopotamus", "confidence": 99.8},
    {"species": "buffalo", "confidence": 0.2},
    {"species": "guineaFowl", "confidence": 0.0}
  ],
  "overlay": "<base64-encoded JPEG of the GradCAM heatmap>",
  "predicted_class": "hippopotamus"
}
```

Try it interactively at `/docs`.

## How GradCAM is implemented

Built from scratch rather than using a library — forward/backward hooks
attached to `model.layer4[-1]` (the last convolutional block, the final
point in the network where spatial information survives), gradients used to
weight each channel's importance for the predicted class, and the weighted
feature maps summed and overlaid on the preprocessed input image.

The forward pass for predictions and the GradCAM heatmap come from a single
backward-enabled pass — no duplicated computation, one request, one
response with both the answer and the explanation.

## A note on what this model is good at

Trained exclusively on **camera trap imagery** — fixed cameras,
motion-triggered, animals at a distance, natural lighting and framing.

It performs well on images with that style. It performs noticeably worse on
close-up, posed wildlife photography — this is a known, documented
limitation, not an oversight. The GradCAM overlay is genuinely useful here:
it visibly shows the model relying on body silhouette in ambiguous cases
(e.g. confusing a clearly-visible hippopotamus for a buffalo), which is the
same failure mode the test-set confusion matrix predicted.

For best results, try images that look like they could come from a camera
trap: animals at a moderate distance, natural outdoor settings, not posed
close-ups.

## Tech

PyTorch · torchvision · OpenCV · FastAPI · Docker