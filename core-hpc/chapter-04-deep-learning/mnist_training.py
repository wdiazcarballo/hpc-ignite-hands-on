#!/usr/bin/env python3
"""
MNIST Training - การ train โมเดลจำแนกตัวเลขเขียนมือ
Chapter 4: Deep Learning on HPC

ตัวอย่างครบวงจร: data loading, model, training loop, evaluation
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """โมเดล CNN อย่างง่ายสำหรับ MNIST"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train หนึ่ง epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(test_loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="MNIST Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("🔢 MNIST Training with PyTorch")
    print("   Chapter 4: Deep Learning on HPC")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load data
    print("\n📥 Loading MNIST dataset...")
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")

    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: SimpleCNN ({total_params:,} parameters)")

    # Training loop
    print("\n🏋️ Training:")
    print("-" * 50)

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")

    total_time = time.time() - total_start

    print("-" * 50)
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Total Time: {total_time:.1f}s")
    print(f"   Avg Time/Epoch: {total_time/args.epochs:.1f}s")

    # Save model (optional)
    # torch.save(model.state_dict(), "mnist_model.pth")
    # print("\n💾 Model saved to mnist_model.pth")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
