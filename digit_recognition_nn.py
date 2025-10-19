from re import X
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Setup
# --------------------------------------------------------

# use cuda if gpu is available
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# parameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# pre-processing function
transform_function = transforms.Compose([
    transforms.ToTensor(),  # convert [0,255] pixels to [0,1] tensor
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1,+1]
])

# --------------------------------------------------------
# Data
# --------------------------------------------------------

# train dataset, disable lazy loading
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform_function,
    download=True
)

# test dataset, disable lazy loading
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform_function,
    download=True
)

# check data size
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=(device.type == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=(device.type == "cuda"))
images, labels = next(iter(train_loader))
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
print(images.shape)  # [64, 1, 28, 28]

# --------------------------------------------------------
# Model
# --------------------------------------------------------

# A simple neutral network to predict whether the digit is 0 or 1
class DigitClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x) # [batch, 28,28] to [batch, 784]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        logit = self.layer3(x)
        return logit

model = DigitClassificationModel().to(device)

# --------------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------------
# Train and Eval Functions
# --------------------------------------------------------

def train_one_epoch(epoch):
  model.train()
  running_loss, correct, total = 0.0, 0, 0

  for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)  # [batch, 10]
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=1)
    running_loss += loss.item() * images.size(0)
    correct += (preds == labels).sum().item()
    total += images.size(0)

  test_loss = running_loss / total
  test_acc = correct / total
  print(f"Train  Loss: {test_loss:.4f} | Train  Accuracy: {test_acc:.4f}")

@torch.inference_mode()
def evaluate():
  model.eval()
  running_loss, correct, total = 0.0, 0, 0

  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)
    
    preds = logits.argmax(dim=1)
    running_loss += loss.item() * images.size(0)
    correct += (preds == labels).sum().item()
    total += images.size(0)
  
  eval_loss = running_loss / total
  eval_acc = correct / total
  print(f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_acc:.4f}")


# --------------------------------------------------------
# Run
# --------------------------------------------------------
for epoch in range(EPOCHS):
  train_one_epoch(epoch)
  evaluate()

# ----------------------------
# Inference
# ----------------------------
@torch.inference_mode()
def sample_predictions(n):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    preds = logits.argmax(dim=1)

    print("\nFirst", n, "GT labels:", labels[:n].tolist())
    print("First", n, "preds     :", preds[:n].tolist())

    img = images[0].squeeze().cpu()
    label = labels[0].item()
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

sample_predictions(5)
