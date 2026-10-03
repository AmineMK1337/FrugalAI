"""Fine-tune ImageNet-pretrained ResNet18 on dataset/train, select best epoch on dataset/val.
Produces resnet18_baseline.pth for quantization_experiment.py. The test set is never touched."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR, NUM_CLASSES, IMAGE_SIZE = "dataset", 2, 224
BATCH_SIZE, EPOCHS, LR = 32, 5, 1e-4
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                              transforms.ToTensor(), norm])
train_dl = DataLoader(datasets.ImageFolder(f"{DATA_DIR}/train", train_tf),
                      batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(datasets.ImageFolder(f"{DATA_DIR}/val", eval_tf), batch_size=BATCH_SIZE)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    print(f"epoch {epoch + 1}/{EPOCHS}  val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet18_baseline.pth")  # plain state_dict
print("Best val acc:", best_acc, "-> saved resnet18_baseline.pth")
