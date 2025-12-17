from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from utils import pgd_attack


def load_data(batch_size=256):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = load_data(batch_size=64)

    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()

    # 阶段1：先冻结特征层，只训练分类头
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=5e-5)

    num_epochs = 100
    best_accuracy = 0.0
    stage2_start_epoch = 50  # 50轮后进入阶段2：微调部分特征层
    for epoch in tqdm(range(num_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{num_epochs}:")

        # 阶段2：50轮后解冻部分特征层进行微调
        if epoch == stage2_start_epoch:
            print("    Enter Stage 2: Fine-tune partial feature layers")
            # 解冻前4层特征层（平衡性能和训练速度）
            for i, (name, param) in enumerate(model.named_parameters()):
                if i < 4:
                    param.requires_grad = True
            # 调低学习率进行微调
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)


        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # 1. 原始样本训练
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total += len(labels)

            # 2. PGD对抗样本训练
            optimizer.zero_grad()
            
            adv_images = pgd_attack(model, images, labels, criterion, epsilon=0.5, alpha=0.05, steps=20)
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            adv_loss.backward()
            optimizer.step()
            
            train_loss += adv_loss.item()
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            train_correct += (adv_predicted == labels).sum().item()
            total += len(labels)

        train_loss /= len(train_loader) * 2
        train_accuracy = 100.0 * train_correct / total
        print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
            print(f"    Learning rate updated to: {param_group['lr']:.6f}")


        model.eval()
        test_loss = 0.0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                total += len(labels)

        test_loss /= len(test_loader)
        test_accuracy = 100.0 * test_correct / total
        print(f"    Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), f'./model/wide_resnet_at.pth')
            best_accuracy = test_accuracy
            print(f"    Best model saved! New best accuracy: {best_accuracy:.2f}%")