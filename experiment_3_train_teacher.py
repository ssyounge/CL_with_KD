"""
train_teacher.py
위치: FACIL/src/approach/train_teacher.py
기능: Teacher 모델(CIFAR-100용 resnet32)을 학습한 뒤, .pth 파일로 저장
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# (A) sys.path 설정 (FACIL/src/ 경로 인식)
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.networks.resnet32 import resnet32   # 이제 "ModuleNotFoundError" 없음

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save-checkpoint', type=str, default='/home/suyoung425/FACIL/checkpoints/T1.pth')
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # (A) Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=2)

    # (B) Teacher model
    net = resnet32(num_classes=100)
    net = net.to(device)

    # (C) Loss / Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # (D) Train loop
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # (optional) test
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss {running_loss:.3f} | TestAcc {acc:.2f}%")

    # (E) Save teacher checkpoint
    os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
    torch.save(net.state_dict(), args.save_checkpoint)
    print(f"Teacher model saved to {args.save_checkpoint}!")

if __name__ == "__main__":
    main()