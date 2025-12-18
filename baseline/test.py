import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# 1. 设备配置：优先使用GPU，无则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 数据预处理与加载
# ResNet50预训练模型要求的标准化参数（ImageNet数据集）
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 训练集增强：随机裁剪、水平翻转 + 标准化；测试集仅resize和标准化
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪并resize到224×224
    transforms.RandomHorizontalFlip(),   # 随机水平翻转（数据增强）
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 加载CIFAR10数据集
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

# 数据加载器
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# CIFAR10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 加载预训练ResNet50并改造分类层
# 加载预训练模型
model = models.resnet50(pretrained=True)

# 冻结主干网络所有参数（仅训练分类层）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层全连接层：原in_features=2048，out_features=10（CIFAR10类别数）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 新的分类层，参数默认可训练（requires_grad=True）

# 将模型移到指定设备
model = model.to(device)

# 4. 训练配置：仅优化分类层参数
criterion = nn.CrossEntropyLoss()  # 分类任务损失函数
# 优化器仅传入fc层的参数（主干参数已冻结，无需优化）
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# 5. 训练与验证函数
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # 训练模式（启用Dropout等）
    train_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # 数据移到设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 + 计算损失 + 反向传播 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item() * inputs.size(0)
            
            # 打印批次信息
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算本轮平均损失
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}')
    
    return model, train_losses

def evaluate_model(model, test_loader):
    model.eval()  # 验证模式（关闭Dropout等）
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算，提升速度
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 取预测概率最大的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

# 6. 执行训练（建议至少训练5-10轮，可根据需求调整）
epochs = 50
model, train_losses = train_model(model, train_loader, criterion, optimizer, epochs)

# 7. 评估模型
accuracy = evaluate_model(model, test_loader)

# 8. 可视化训练损失
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('ResNet50 Fine-tuning (Only FC Layer) on CIFAR10 - Loss Curve')
plt.grid(True)
plt.show()

# 9. 保存微调后的模型（可选）
torch.save(model.state_dict(), 'resnet50_cifar10_finetuned_fc.pth')
print("模型已保存为 resnet50_cifar10_finetuned_fc.pth")