import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# 1. Load dataset (use only a small subset: 10 samples)
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full MNIST training set
    full_train = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)

    # Sample only 10 random indices for few-shot training
    small_indices = torch.randperm(len(full_train))[:10]
    small_train = Subset(full_train, small_indices)

    # Load standard MNIST test set
    test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(small_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    return train_loader, test_loader

# 2. Define the CNN model
class MNISTModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# 3. Evaluation function: compute accuracy and average loss
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {'acc': correct / total, 'loss': loss_sum / total}

# 4. Standard training loop (no LLM guidance)
def train_baseline():
    train_loader, test_loader = get_dataloaders()
    model = MNISTModel(dropout=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader)
        val_metrics = evaluate(model, test_loader)

        print(f"Epoch {epoch}: Train Acc = {train_metrics['acc']:.4f}, Val Acc = {val_metrics['acc']:.4f}")

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics
        })

    # Plot validation accuracy over epochs
    epochs = [h['epoch'] for h in history]
    val_accs = [h['val']['acc'] for h in history]

    plt.plot(epochs, val_accs, label="Val Acc (No LLM)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (Baseline, 10 samples)")
    plt.grid(True)
    plt.legend()
    plt.savefig("mnist_baseline_10samples.png")
    plt.show()

if __name__ == '__main__':
    train_baseline()
