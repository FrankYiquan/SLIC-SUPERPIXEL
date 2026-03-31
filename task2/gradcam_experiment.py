import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

IMAGE_PATH = "images/input/horse.jpg"
OUTPUT_DIR = "images/output"
TARGET_CLASS = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Load model 
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()


# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
input_tensor = normalize(input_tensor)

# visualization image
rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
rgb_img = np.clip(rgb_img, 0, 1)

# Predict class
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

if TARGET_CLASS is None:
    TARGET_CLASS = pred_class

print(f"Using class index: {TARGET_CLASS}")

targets = [ClassifierOutputTarget(TARGET_CLASS)]


def get_cam(cam_obj, tensor):
    cam = cam_obj(input_tensor=tensor, targets=targets)[0]
    cam = np.array(cam)

    if cam.ndim > 2:
        cam = cam.squeeze()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.astype(np.float32)

    return cam

def save_image(img, name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.imsave(path, img)
    print(f"Saved: {path}")

# Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = get_cam(cam, input_tensor)
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

save_image(cam_image, "gradcam_layer4.png")

# Vanilla Backprop
input_tensor_vb = input_tensor.clone().requires_grad_(True)

output = model(input_tensor_vb)
loss = output[0, TARGET_CLASS]
loss.backward()

vanilla_grad = input_tensor_vb.grad[0].permute(1, 2, 0).detach().numpy()
vanilla_grad = (vanilla_grad - vanilla_grad.min()) / (vanilla_grad.max() - vanilla_grad.min() + 1e-8)

save_image(vanilla_grad, "vanilla_backprop.png")

# Guided Backprop 
device = torch.device("cpu")
gb_model = GuidedBackpropReLUModel(model=model, device=device)

gb = gb_model(input_tensor, target_category=TARGET_CLASS)

# Normalize for visualization
gb = (gb - gb.min()) / (gb.max() - gb.min() + 1e-8)

save_image(gb, "guided_backprop.png")

guided_grad_cam = gb * grayscale_cam[..., np.newaxis]

# Normalize
guided_grad_cam = (guided_grad_cam - guided_grad_cam.min()) / (guided_grad_cam.max() - guided_grad_cam.min() + 1e-8)

save_image(guided_grad_cam, "guided_gradcam.png")


# Layer comparison
layers = {
    "relu": model.relu,
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4,
}

for name, layer in layers.items():
    cam = GradCAM(model=model, target_layers=[layer])
    grayscale_cam = get_cam(cam, input_tensor)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    save_image(cam_image, f"gradcam_{name}.png")

# Rotation
rotated_img = image.rotate(30)
rot_tensor = transform(rotated_img).unsqueeze(0)
rot_tensor = normalize(rot_tensor)

rgb_rot = np.array(rotated_img.resize((224, 224))).astype(np.float32) / 255.0

cam = GradCAM(model=model, target_layers=[target_layer])
rot_cam = get_cam(cam, rot_tensor)

rot_cam_img = show_cam_on_image(rgb_rot, rot_cam, use_rgb=True)
save_image(rot_cam_img, "rotated_gradcam.png")

# Noise
noisy_tensor = input_tensor + 0.05 * torch.randn_like(input_tensor)

cam = GradCAM(model=model, target_layers=[target_layer])
noise_cam = get_cam(cam, noisy_tensor)

noise_cam_img = show_cam_on_image(rgb_img, noise_cam, use_rgb=True)
save_image(noise_cam_img, "noisy_gradcam.png")

print("All outputs saved to images/output/")