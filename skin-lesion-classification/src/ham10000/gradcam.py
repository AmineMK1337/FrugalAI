from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
from ham10000.utils import load_config
from ham10000.data import build_dataloaders
from ham10000.evaluate import load_model
import random


def prepare_image(path, img_size=224):
    img = Image.open(path).convert('RGB').resize((img_size, img_size))
    rgb_img = np.array(img, dtype=np.float32) / 255.0
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = normalize(rgb_img).unsqueeze(0)
    return rgb_img, input_tensor

def save_gradcam(rgb_img, visualization, true_cls, pred_cls, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(rgb_img)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(visualization)
    ax2.set_title(f'True: {true_cls} | Pred: {pred_cls}')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    
def collect_samples(model, val_subset, device, n_each=5):
    dataset = val_subset.dataset
    df = dataset.data
    class_to_idx = dataset.class_to_idx
    image_paths = dataset.image_paths
    
    correct, incorrect = [], []
    model.eval()
    indices = list(val_subset.indices)
    random.seed(42)            # reproducible
    random.shuffle(indices)
    
    for row_index in indices:
        row = df.iloc[row_index]
        image_id = row['image_id']
        true_idx = class_to_idx[row['dx']]
        path = image_paths.get(image_id)
        if path is None:
            continue
        _, input_tensor = prepare_image(path)
        with torch.no_grad():
            output = model(input_tensor.to(device))
            pred_idx = int(output.argmax(dim=1).item())
        
        sample = {'path': path,
                  'true_idx': true_idx,
                  'pred_idx': pred_idx}
        if pred_idx == true_idx and len(correct) < n_each:
            correct.append(sample)
        elif pred_idx != true_idx and len(incorrect) < n_each:
            incorrect.append(sample)
        
        if len(correct) >= n_each and len(incorrect) >= n_each:
            break
        
    return correct, incorrect

def run_gradcam_on_samples(model, samples, idx_to_class, prefix, device):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    for i, sample in enumerate(samples, start=1):
        rgb_img, input_tensor = prepare_image(sample['path'])
        input_tensor = input_tensor.to(device)
        
        targets = [ClassifierOutputTarget(sample['pred_idx'])]
        grayscale_cam = cam(input_tensor, targets=targets)[0]
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        true_cls = idx_to_class[sample['true_idx']]
        pred_cls = idx_to_class[sample['pred_idx']]
        save_gradcam(rgb_img, visualization, true_cls, pred_cls, f'gradcam_{prefix}_{i}.png')

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for HAM10000 classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    _, val_loader, class_to_idx = build_dataloaders(config)
    val_subset = val_loader.dataset
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = load_model(config, args.checkpoint, device)

    correct, incorrect = collect_samples(model, val_subset, device, n_each=5)
    print(f"Found {len(correct)} correct and {len(incorrect)} incorrect samples")

    run_gradcam_on_samples(model, correct, idx_to_class, "correct", device)
    run_gradcam_on_samples(model, incorrect, idx_to_class, "incorrect", device)


if __name__ == "__main__":
    main()