import numpy as np
import cv2

def is_blurry(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score

def crop_timestamp(img, crop_ratio=0.9):
    h = img.shape[0]
    return img[:int(h * crop_ratio), :, :]

def apply_clahe(img, is_night, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    if is_night:
        a = np.full_like(a, 128)
        b = np.full_like(b, 128)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def preprocess_image(img, is_night=False, crop_timestamp_bar=False):
    if crop_timestamp_bar:
        img = crop_timestamp(img)
    img = apply_clahe(img, is_night=is_night)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    return img