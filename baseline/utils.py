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


if __name__ == '__main__':
    train_loader, test_loader = load_data()

    dataiter = iter(test_loader)
    images, lables = next(dataiter)
    print(images.shape)
