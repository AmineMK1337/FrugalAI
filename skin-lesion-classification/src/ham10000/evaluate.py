import torch
from ham10000.models import build_model
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import argparse
from ham10000.utils import load_config
from ham10000.data import build_dataloaders


def get_predictions(model, loader, device):
    """Run inference over the loader, returning (all_labels, all_preds) lists."""
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels

def load_model(config, checkpoint_path, device):
    """Rebuild the model architecture and load saved weights from checkpoint."""
    
    model = build_model(config.num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model

def save_confusion_matrix(all_labels, all_preds, class_names, path, normalize=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds,
        display_labels=class_names,
        xticks_rotation=45,
        normalize=normalize,
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate HAM10000 Classifier')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default="best_model.pth")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')

    _, val_loader, class_to_idx = build_dataloaders(config)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    model = load_model(config, args.checkpoint, device)
    all_preds, all_labels = get_predictions(model, val_loader, device)

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    suffix = "weighted" if config.use_class_weights else "baseline"
    report_path = f"classification_report_{suffix}_aug.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved {report_path}")
    
    save_confusion_matrix(all_labels, all_preds, class_names,
                      f"confusion_matrix_{suffix}_aug.png")
    save_confusion_matrix(all_labels, all_preds, class_names,
                        f"confusion_matrix_{suffix}_aug_normalized.png", normalize="true")

if __name__ == "__main__":
    main()