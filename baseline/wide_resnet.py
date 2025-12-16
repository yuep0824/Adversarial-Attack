import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# -------------------------- 1. 配置参数 --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_STAGE1 = 10  # 第一阶段：训练分类头
EPOCHS_STAGE2 = 20  # 第二阶段：微调部分层
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10
SAVE_PATH = "best_wideresnet_cifar10.pth"

# -------------------------- 2. 数据预处理与加载 --------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Wide ResNet预训练模型输入为224x224
    transforms.RandomHorizontalFlip(),  # 数据增强：随机水平翻转
    transforms.RandomCrop(224, padding=4),  # 数据增强：随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet归一化均值
                         std=[0.229, 0.224, 0.225])   # ImageNet归一化标准差
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------------------- 3. 加载预训练Wide ResNet模型 --------------------------
# 加载预训练的Wide ResNet50_2（宽度因子2，深度50）
model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)

# 修改最后一层全连接层：适配CIFAR-10的10分类
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# -------------------------- 4. 定义损失函数和优化器 --------------------------
criterion = nn.CrossEntropyLoss()

# -------------------------- 5. 训练和验证函数 --------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch, stage):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Stage {stage} Epoch {epoch+1} [Train]")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Validation")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(loader.dataset)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# -------------------------- 6. 分阶段微调 --------------------------
best_val_acc = 0.0

# 阶段1：冻结特征提取层，只训练分类头（fc层）
print("="*50)
print("Stage 1: Train only the classification head (fc layer)")
print("="*50)

# 冻结所有层，只解冻fc层
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True  # 只训练最后一层

# 优化器只更新fc层的参数
optimizer_stage1 = optim.AdamW(model.fc.parameters(), 
                               lr=LEARNING_RATE_STAGE1, 
                               weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS_STAGE1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage1, criterion, epoch, stage=1)
    val_loss, val_acc = validate(model, test_loader, criterion)
    
    print(f"Stage 1 Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Stage 1 Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Best model saved! New best val acc: {best_val_acc:.2f}%")

# 阶段2：解冻部分层，微调（解冻前几层+分类头）
print("\n" + "="*50)
print("Stage 2: Fine-tune part of the feature layers + classification head")
print("="*50)

# 解冻前4层（可根据需求调整解冻层数）
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 4:  # 解冻前4层
        param.requires_grad = True
    # 保持其他层冻结（也可全部解冻，学习率调低即可）

# 优化器更新所有可训练参数
optimizer_stage2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=LEARNING_RATE_STAGE2,
                               weight_decay=WEIGHT_DECAY)
# 学习率调度器：每10个epoch学习率减半
scheduler = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=10, gamma=0.5)

for epoch in range(EPOCHS_STAGE2):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_stage2, criterion, epoch, stage=2)
    val_loss, val_acc = validate(model, test_loader, criterion)
    
    print(f"Stage 2 Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Stage 2 Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Best model saved! New best val acc: {best_val_acc:.2f}%")
    
    scheduler.step()


# -------------------------- 7. 加载最佳模型并评估 --------------------------
print("\n" + "="*50)
print("Final Evaluation on Test Set")
print("="*50)

# 加载最佳模型
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 按类别统计准确率
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


overall_acc = 100. * correct / total
print(f'Overall Test Accuracy of the model on the 10000 test images: {overall_acc:.2f}%')
print("\nClass-wise Accuracy:")
for i in range(10):
    acc = 100. * class_correct[i] / class_total[i]
    print(f'{classes[i]}: {acc:.2f}%')