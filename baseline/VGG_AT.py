from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data, pgd_attack


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    

if __name__ == '__main__':
    train_loader, test_loader = load_data(batch_size=256)

    model = VGG().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    # 训练模型
    num_epochs = 200
    model.train()
    best_accuracy = 0.0
    for epoch in tqdm(range(num_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{num_epochs}:")
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
            torch.save(model.state_dict(), f'./model/vgg19_at.pth')
            best_accuracy = test_accuracy
