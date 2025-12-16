from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.classifier(x)
        
        return x
    

if __name__ == '__main__':
    train_loader, test_loader = load_data(batch_size=256)

    model = CNN().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    num_epochs = 200
    best_accuracy = 0.0
    for epoch in tqdm(range(num_epochs), desc="Training"):
        print(f"Epoch {epoch+1}/{num_epochs}:")

        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total += len(labels)

        train_loss /= len(train_loader)
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
            torch.save(model.state_dict(), f'./model/cnn.pth')
            best_accuracy = test_accuracy