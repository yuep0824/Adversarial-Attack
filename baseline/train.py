from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data

from baseline.models import CNN, VGG, VisionTransformer
from baseline.models import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet


def get_model(model_name, num_classes=10):
    if model_name == 'cnn':
        model = CNN(num_classes=num_classes)
    elif model_name == 'vgg19':
        model = VGG(num_classes=num_classes)
    elif model_name == 'vit':
        model = VisionTransformer(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes)
    elif model_name == 'wide_resnet':
        model = wide_resnet(num_classes=num_classes)
    
    return model


if __name__ == "__main__":
    train_loader, test_loader = load_data(batch_size=256)

    model_name = 'cnn'  # 可选：cnn, vgg19, vit, resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet
    model = get_model(model_name, num_classes=10).cuda()

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
            torch.save(model.state_dict(), f'./model/{model_name}.pth')
            best_accuracy = test_accuracy