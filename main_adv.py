import torch
import torch.nn as nn
from torchvision import transforms

import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


import sys
sys.path.append(".")

from attacks.FGSM import fgsm_attack
from attacks.PC_I_FGSM import pc_i_fgsm_attack
from attacks.PGD import pgd_attack, multi_restart_pgd_attack, nes_pgd_attack, cw_pgd_hybrid_attack
from attacks.DeepFool import deepfool_attack
from attacks.CW import CW
from attacks.boundary_attack import BoundaryAttack

from baseline.models import CNN, VGG, ViT
from baseline.models import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet
from baseline.models1 import densenet121, densenet161, densenet169, googlenet, inception_v3, mobilenet_v2, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from baseline.models2 import cifar10_resnet56, cifar10_vgg16_bn, cifar10_mobilenetv2_x1_0, cifar10_shufflenetv2_x2_0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    model.load_state_dict(torch.load(f'./model/{model_name}_at.pth', map_location=device))
    model = model.to(device)
    model.eval()

    return model


def get_new_model(model_name, num_classes=10):
    if model_name == 'cifar10_resnet56':
        model = cifar10_resnet56()
        model_path = './model/state_dicts/cifar10_resnet56-187c023a.pt'
    elif model_name == 'cifar10_vgg16_bn':
        model = cifar10_vgg16_bn()
        model_path = './model/state_dicts/cifar10_vgg16_bn-6ee7ea24.pt'
    elif model_name == 'cifar10_mobilenetv2_x1_0':
        model = cifar10_mobilenetv2_x1_0()
        model_path = './model/state_dicts/cifar10_mobilenetv2_x1_0-fe6a5b48.pt'
    elif model_name == 'cifar10_shufflenetv2_x2_0':
        model = cifar10_shufflenetv2_x2_0()
        model_path = './model/state_dicts/cifar10_shufflenetv2_x2_0-ec31611c.pt'
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes)
        model_path = './model/state_dicts/densenet121.pt'
    elif model_name == 'densenet161':  
        model = densenet161(num_classes=num_classes)
        model_path = './model/state_dicts/densenet161.pt'
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes)
        model_path = './model/state_dicts/densenet169.pt'
    elif model_name == 'googlenet':
        model = googlenet(num_classes=num_classes)
        model_path = './model/state_dicts/googlenet.pt'
    elif model_name == 'inception_v3':
        model = inception_v3(num_classes=num_classes)
        model_path = './model/state_dicts/inception_v3.pt'
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2(num_classes=num_classes)
        model_path = './model/state_dicts/mobilenet_v2.pt'
    elif model_name == 'vgg11_bn':
        model = vgg11_bn(num_classes=num_classes)
        model_path = './model/state_dicts/vgg11_bn.pt'
    elif model_name == 'vgg13_bn':
        model = vgg13_bn(num_classes=num_classes)
        model_path = './model/state_dicts/vgg13_bn.pt'
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(num_classes=num_classes)
        model_path = './model/state_dicts/vgg16_bn.pt'
    elif model_name == 'vgg19_bn':
        model = vgg19_bn(num_classes=num_classes)
        model_path = './model/state_dicts/vgg19_bn.pt'

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    adv_attack = 'pgd'  # 可选：fgsm, pc_i_fgsm, pgd, deepfool, cw, boundary, nes_pgd_attack, multi_restart_pgd, cw_pgd_hybrid_attack
    # attack_model =  'resnet34'  # cnn, vgg19, vit, resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet
    # model = get_model(attack_model, num_classes=10)

    attack_model = 'cifar10_resnet56' # cifar10_vgg16_bn, cifar10_mobilenetv2_x1_0, cifar10_shufflenetv2_x2_0, densenet121, densenet161, densenet169, googlenet, inception_v3, mobilenet_v2, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
    model = get_new_model(attack_model, num_classes=10)

    if attack_model == 'wide_resnet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        inv_transform = transforms.Compose([
            transforms.Normalize((-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616), (1.0/0.2470, 1.0/0.2435, 1.0/0.2616)),
            transforms.Resize((32, 32))
        ])
    else :
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        inv_transform = transforms.Compose([
            transforms.Normalize((-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616), (1.0/0.2470, 1.0/0.2435, 1.0/0.2616))
        ])

    clean_folder = "./data/cifar10_clean_500/"
    clean_image_folder = os.path.join(clean_folder, 'images/')
    clean_label_path = os.path.join(clean_folder, 'label.txt')

    label_dict = {}
    with open(clean_label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                image_name, label = line.split()
                label_dict[image_name] = int(label)

    adv_folder = f"./data/cifar10_{adv_attack}_{attack_model}_500/"
    adv_image_folder = os.path.join(adv_folder, 'images/')
    adv_label_path = os.path.join(adv_folder, 'label.txt')
    os.makedirs(adv_image_folder, exist_ok=True)
    shutil.copy2(clean_label_path, adv_label_path)


    # 对抗攻击应用
    for image_name in tqdm(os.listdir(clean_image_folder), desc="Adversarial Processing"):
        img_path = os.path.join(clean_image_folder, image_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        input = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        input = transform(input).unsqueeze(0).float().requires_grad_(True)
        # 新增：输入移到CUDA
        input = input.to(device)

        label = torch.tensor([label_dict[image_name]])
        # 新增：标签移到CUDA
        label = label.to(device)

        # 原有攻击逻辑不变...
        if adv_attack == 'fgsm':
            adversarial_tensor = fgsm_attack(model, input, label, epsilon=0.4)
        elif adv_attack == 'pc_i_fgsm':
            adversarial_tensor = pc_i_fgsm_attack(model, input, label, epsilon=0.1)
        elif adv_attack == 'pgd':
            adversarial_tensor = pgd_attack(model, input, label, epsilon=0.8, alpha=0.1, num_iterations=20, device=device)
        elif adv_attack == 'multi_restart_pgd':
            adversarial_tensor = multi_restart_pgd_attack(model, input, label, epsilon=0.5, alpha=0.05, num_iterations=20)
        elif adv_attack == 'nes_pgd_attack':
           adversarial_tensor = nes_pgd_attack(model, input, label, epsilon=0.8, alpha=0.1, num_iterations=20)
        elif adv_attack == 'cw_pgd_hybrid_attack':
            adversarial_tensor = cw_pgd_hybrid_attack(model, input, label, epsilon=0.8, alpha=0.1, num_iterations=30, kappa=0.2)
        elif adv_attack == 'deepfool':
            adversarial_tensor, _ = deepfool_attack(model, input, label, overshoot=0.1, max_iter=50)
        elif adv_attack == 'cw':
            adversarial_tensor = CW(model).attack(input, label)
        elif adv_attack == 'boundary':
            attacker = BoundaryAttack(model, max_k=100, delta=0.1)
            perturb_list, _, _ = attacker.attack_untarget(input, label)
            adversarial_tensor = perturb_list[-1]
        else:
            adversarial_tensor = input  # 未知攻击方式，返回原图

        adv_output = model(adversarial_tensor)
        _, adv_label = torch.max(adv_output.data, 1)
        print(f"GT Label: {label}, Adv Label is {adv_label}")


        adv_img = adversarial_tensor.detach().squeeze(0).cpu()
        adv_img = inv_transform(adv_img).permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, 0.0, 1.0) * 255.0
        adv_img = adv_img.astype(np.uint8)
        adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join(adv_image_folder, image_name)
        cv2.imwrite(save_path, adv_img)

    # 对抗攻击SSIM检测
    ssim_scores = []
    for image_name in os.listdir(adv_image_folder):
        adv_path = os.path.join(adv_image_folder, image_name)
        clean_path = os.path.join(clean_image_folder, image_name)
        adv_image = cv2.imread(adv_path)
        clean_image = cv2.imread(clean_path)
        target_size = (32, 32)
        adv_image = cv2.resize(adv_image, target_size, interpolation=cv2.INTER_LANCZOS4)
        clean_image = cv2.resize(clean_image, target_size, interpolation=cv2.INTER_LANCZOS4)
        adv_image_rgb = cv2.cvtColor(adv_image, cv2.COLOR_BGR2RGB)
        clean_image_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        adv_image_norm = adv_image_rgb / 255.0
        clean_image_norm = clean_image_rgb / 255.0
        

        ssim_score = ssim(clean_image_norm, adv_image_norm, channel_axis=-1, data_range=1.0)
        ssim_scores.append(ssim_score)

    # 统计并打印整体结果
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        print("\n==================== 统计结果 ====================")
        print(f"有效计算的样本数量：{len(ssim_scores)}")
        print(f"平均SSIM值：{avg_ssim:.6f}")
        print(f"SSIM值标准差：{std_ssim:.6f}")
        print(f"最小SSIM值：{np.min(ssim_scores):.6f}")
        print(f"最大SSIM值：{np.max(ssim_scores):.6f}")