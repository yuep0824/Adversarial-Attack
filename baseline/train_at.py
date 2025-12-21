from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data, pgd_attack, fgsm_attack
from models import CNN, VGG, ViT
from models import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet


def get_model(model_name, num_classes=10):
    if model_name == 'cnn':
        model = CNN(num_classes=num_classes)
    elif model_name == 'vgg19':
        model = VGG(num_classes=num_classes)
    elif model_name == 'vit':
        model = ViT(num_classes=num_classes)
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

    model_name = 'vit'  # 可选：cnn, vgg19, vit, resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet
    model = get_model(model_name, num_classes=10).cuda()
    model.load_state_dict(torch.load(f'./model/{model_name}_pre_at.pth'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    num_epochs = 200
    num_adv_epochs = 100
    adv_sample_ratio = 0.1
    best_accuracy = 0.0
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        adv_loss = 0.0
        adv_correct = 0
        adv_total = 0

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
            if epoch >= num_adv_epochs:  
                # 随机选择指定比例的样本索引
                batch_size = images.shape[0]
                adv_indices = torch.randperm(batch_size)[:int(batch_size * adv_sample_ratio)]
                selected_images = images[adv_indices]
                selected_labels = labels[adv_indices]
                
                # adv_images = pgd_attack(model, selected_images, selected_labels, criterion, epsilon=0.2, alpha=0.04, steps=5)
                adv_images = fgsm_attack(model, selected_images, selected_labels, criterion, epsilon=0.5)
                adv_outputs = model(adv_images)
                loss = criterion(adv_outputs, selected_labels)
                loss.backward()
                optimizer.step()

                adv_loss += loss.item()
                _, adv_predicted = torch.max(adv_outputs.data, 1)
                adv_correct += (adv_predicted != selected_labels).sum().item()
                adv_total += len(selected_labels)
            

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / total

        print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        if epoch >= num_adv_epochs:
            adv_loss /= len(train_loader)
            adv_accuracy = 100.0 * adv_correct / adv_total
            print(f"    Adversarial Loss: {adv_loss:.4f}, Adversarial Accuracy: {adv_accuracy:.2f}%")

        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8


        # 评估模型
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