import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        score = output[0, class_idx]
        score.backward()

        alpha = self.gradients[0].mean(dim=(1,2))

        weighted_maps = self.activations[0] * alpha[:, None, None]
        cam = weighted_maps.sum(dim=0)
        cam=F.relu(cam)

        cam=cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx, output
    
def overlay_heatmap(original_img, cam, alpha=0.4):
    h,w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w,h))
    heatmap = np.uint8(255*cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)

    return overlaid
    