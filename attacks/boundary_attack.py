import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
            nn.Linear(64*8*8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*8*8)
        x = self.classifier(x)
        
        return x


class BoundaryAttack:
    def __init__(self, model, max_k, delta):
        self.model = model
        self.max_k = max_k

    def get_diff(self, x, y):
        return torch.norm(x - y, dim=(2, 3), keepdim=True)
    
    def forward_perturbation(self, epsilon, adversarial_image, origin_image):
        perturb = origin_image - adversarial_image
        perturb = epsilon * perturb.float()
        return perturb
    
    def calculate_vdot(self, x, y):
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        return torch.sum(x_flat * y_flat, dim=1)

    def get_perturb(self, delta, adversarial_image, origin_image):
        # step1: generate random perturb
        perturb = torch.randn(1, 3, 32, 32)
        perturb /= torch.norm(perturb, dim=(2, 3), keepdim=True)
        perturb *= delta * torch.mean(self.get_diff(origin_image, adversarial_image))

        # step2: project the perturb onto a sphere around the original image
        diff = (origin_image - adversarial_image).float()
        diff /= self.get_diff(origin_image, adversarial_image)  # diff: a unit vector in diff direction
        perturb -= self.calculate_vdot(perturb, diff) * diff

        # 截断，防止溢出
        lower_bound = torch.zeros((1, 3, 32, 32)).float()
        upper_bound = torch.ones((1, 3, 32, 32)).float()
        perturb = torch.clamp(perturb, lower_bound - adversarial_image, upper_bound - adversarial_image)
        return perturb

    def attack_untarget(self, origin_image, origin_label):
        # 初始化对抗样本
        adversarial_image = torch.rand((1, 3, 32, 32))

        # 二分搜索找到决策边界
        l, r, epsilon = 0.0, 1.0, 0.5
        template_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
        while True:
            template_outputs = self.model(template_image)
            _, template_predict = torch.max(template_outputs, 1)
            if template_predict == origin_label:
                r = (l + r) / 2.0
            else:
                # 验证是否满足二分结束条件, 边界值且符合对抗要求
                if abs(l - r) < 1e-4: 
                    adversarial_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
                    break
                l = (l + r) / 2.0
            
            epsilon = (l + r) / 2.0
            template_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)

        # 正交步骤和步长调整
        k = 0
        delta = 0.2
        perturb_list, epsilons, deltas = [], [], []
        perturb_list.append(adversarial_image)
        epsilons.append(epsilon)
        while k < self.max_k:
            # 生成正交扰动 step1
            while True:
                test_images = []
                correct = 0.0
                total = 10
                for _ in range(total):
                    perturb = self.get_perturb(delta, adversarial_image, origin_image)
                    test_images.append(adversarial_image + perturb)
                # 测试正交扰动, 如果正交扰动成功，更新对抗样本
                for data in test_images:
                    outputs = self.model(data)
                    _, predict = torch.max(outputs, 1)
                    correct += (predict != origin_label)
                score = correct / total

                if score < 0.4:
                    delta *= 0.9
                elif score > 0.7: 
                    delta /= 0.99
                    for data in test_images:
                        outputs = self.model(data)
                        _, predict = torch.max(outputs, 1)
                        if predict != origin_label:
                            adversarial_image = data
                            break
                    break
                else:
                    pass
            
            # step3
            steps = 0
            while True:
                steps += 1
                test_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
                outputs = self.model(test_image)
                _, predict = torch.max(outputs, 1)
                if predict != origin_label:
                    adversarial_image = test_image
                    epsilon /= 0.99
                    break
                elif steps > 100:
                    break
                else:
                    epsilon *= 0.9

            perturb_list.append(adversarial_image)
            epsilons.append(epsilon)
            deltas.append(delta)
            k = k + 1

        return perturb_list, epsilons, deltas
           
    def attack_target(self, origin_image, adversarial_image, adversarial_label):
        # 二分搜索找到决策边界
        l, r, epsilon = 0.0, 1.0, 0.5
        template_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
        while True:
            template_outputs = self.model(template_image)
            _, template_predict = torch.max(template_outputs, 1)
            if template_predict != adversarial_label:
                r = (l + r) / 2.0
            else:
                # 验证是否满足二分结束条件, 边界值且符合对抗要求
                if abs(l - r) < 1e-4: 
                    adversarial_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
                    break
                l = (l + r) / 2.0
            
            epsilon = (l + r) / 2.0
            template_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)

        # 正交步骤和步长调整
        k = 0
        delta = 0.2
        perturb_list = []
        perturb_list.append(adversarial_image)
        while k < self.max_k:
            # step1: 生成正交扰动 
            while True:
                test_images = []
                correct = 0.0
                total = 10
                for _ in range(total):
                    perturb = self.get_perturb(delta, adversarial_image, origin_image)
                    test_images.append(adversarial_image + perturb)
                # 测试正交扰动, 如果正交扰动成功，更新对抗样本
                for data in test_images:
                    outputs = self.model(data)
                    _, predict = torch.max(outputs, 1)
                    correct += (predict == adversarial_label)
                score = correct / total

                if score < 0.4:
                    delta *= 0.9
                elif score > 0.7: 
                    delta /= 0.99
                    for data in test_images:
                        outputs = self.model(data)
                        _, predict = torch.max(outputs, 1)
                        if predict == adversarial_label:
                            adversarial_image = data
                            break
                    break
                else:
                    pass

            # step2
            steps = 0
            while True:
                steps += 1
                test_image = adversarial_image + self.forward_perturbation(epsilon, adversarial_image, origin_image)
                outputs = self.model(test_image)
                _, predict = torch.max(outputs, 1)
                if predict == adversarial_label:
                    adversarial_image = test_image
                    epsilon /= 0.99
                    break
                elif steps > 100:
                    break
                else:
                    epsilon *= 0.9

            perturb_list.append(adversarial_image)
            k = k + 1
            print(k)
        return perturb_list


if __name__ == '__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(root="./data/", transform=transform_test, train=False, download=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    inv_transform = transforms.Normalize((-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616), (1.0/0.2470, 1.0/0.2435, 1.0/0.2616))

    model = CNN()
    model.load_state_dict(torch.load("CIFAR10-CNN.pth"))
    model.eval()

    attacker = BoundaryAttack(model, transform, max_k=1005, delta=0.1)

    '''
    # 计算攻击后的准确率
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="Evaluating attack"):
        images, labels = images, LabeledScale()
        images = inv_transform(images)
        perturbed_images = attacker.attack_untarget(images, labels)
        outputs = model(transform(perturbed_images))
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels)

        if total > 500:
            break
    print(f'Accuracy of the model after attack: {100 * correct / total:.2f}%')
    '''
    
    # 选择一个样本进行攻击并可视化
    original_image, original_label = next(iter(test_loader))
    original_image = inv_transform(original_image)
    adversarial_image, adversarial_label = next(iter(test_loader))
    while adversarial_label == original_label:
        adversarial_image, adversarial_label = next(iter(test_loader))

    # Boundary Attack 
    # adversarial_list, epsilons, deltas = attacker.attack_untarget(original_image, original_label)
    adversarial_list = attacker.attack_target(original_image, adversarial_image, adversarial_label)
    
    # mse = F.mse_loss(original_image, adversarial_list[-1])
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(original_image.squeeze().permute(1, 2, 0))
    # axes[0].set_title('Initial Image')
    # axes[0].axis('off')
    # axes[1].imshow(adversarial_list[-1].squeeze().permute(1, 2, 0))
    # axes[1].set_title(f'Adversarial Image \n MSE: {mse:.3e}')
    # axes[1].axis('off')
    # plt.title('Untarget Boundary Attack')
    # plt.show()

    mse = F.mse_loss(original_image, adversarial_list[-1])
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(original_image.squeeze().permute(1, 2, 0))
    axes[0].set_title('Initial Image')
    axes[0].axis('off')
    axes[1].imshow(adversarial_image.squeeze().permute(1, 2, 0))
    axes[1].set_title(f'Target Image')
    axes[1].axis('off')
    axes[2].imshow(adversarial_list[-1].squeeze().permute(1, 2, 0))
    axes[2].set_title(f'Adversarial Image \n MSE: {mse:.3e}')
    axes[2].axis('off')
    plt.show()
    
    # 迭代可视化
    # show_list = [0, 4, 10, 30, 60, 100, 200, 450]
    # fig, axes = plt.subplots(1, len(show_list)+2)
    # axes[0].imshow(original_image.squeeze().permute(1, 2, 0))
    # axes[0].set_title(f'Initial Image \n Label: {original_label.item()}')
    # axes[0].axis('off')
    # axes[1].imshow(adversarial_image.squeeze().permute(1, 2, 0))
    # axes[1].set_title(f'Target Image \n Label: {adversarial_label.item()}')
    # axes[1].axis('off')
    # for i in range(len(show_list)):
    #     output = model(transform(adversarial_list[i]))
    #     _, predict = torch.max(output, 1)
    #     mse = F.mse_loss(original_image, adversarial_list[i])
    #     axes[i+2].imshow(adversarial_list[show_list[i]].squeeze().permute(1, 2, 0))
    #     axes[i+2].set_title(f'Iter: {show_list[i]}times \n Predict: {predict.item()} \n MSE: {mse:.4e}')
    #     axes[i+2].axis('off')
    # plt.show()

    # x = range(1, len(deltas)+1)
    # plt.plot(x, deltas)
    # plt.title('Convergence of Delta')
    # plt.xlabel('Iteration k')
    # plt.ylabel('Delta Value')
    # plt.ylim(0, 1)

    # plt.grid()
    # plt.show()


    # x = range(1, len(epsilons)+1)
    # plt.plot(x, epsilons)
    # plt.title('Convergence of Epsilon')
    # plt.xlabel('Iteration k')
    # plt.ylabel('Epsilon Value')
    # plt.ylim(0, 1)

    # plt.grid()
    # plt.show()

