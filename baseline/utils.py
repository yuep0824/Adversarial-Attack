import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1, 1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader



def fgsm_attack(model, input_image, label, criterion, epsilon=0.2):
    adv_images = input_image.clone().detach().to(input_image.device)
    adv_images.requires_grad = True

    model.eval()

    output = model(adv_images)
    loss = criterion(output, label)
    model.zero_grad()
    loss.backward()

    gradient = adv_images.grad.data.sign()
    adv_images = torch.clamp(adv_images + epsilon * gradient, input_image.min(), input_image.max())

    return adv_images.detach()



def pgd_attack(model, images, labels, criterion, epsilon=0.01, alpha=0.002, steps=10, clamp_min=0.0, clamp_max=1.0):
    adv_images = images.clone().detach().to(images.device)
    adv_images.requires_grad = True

    delta = torch.zeros_like(adv_images).to(images.device)

    for _ in range(steps):
        outputs = model(adv_images + delta)
        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        
        delta = delta + alpha * torch.sign(adv_images.grad.data)
        delta = torch.clamp(delta, -epsilon, epsilon)
        adv_images.grad.zero_()

    adv_images = images + delta
    adv_images = torch.clamp(adv_images, clamp_min, clamp_max)
    
    return adv_images.detach()


if __name__ == '__main__':
    train_loader, test_loader = load_data()

    dataiter = iter(test_loader)
    images, lables = next(dataiter)
    print(images.shape)
