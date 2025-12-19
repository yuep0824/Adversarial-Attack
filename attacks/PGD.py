import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(model, input_image, label, epsilon, alpha, num_iterations):
    model.eval()

    adv_image = input_image.clone()
    for _ in range(num_iterations):
        model.zero_grad()

        adv_image = adv_image.clone().detach().requires_grad_(True)
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, label)
            
        loss.backward()
        gradient = adv_image.grad.data.sign()
        adv_image = torch.clamp(adv_image + alpha * gradient, input_image - epsilon, input_image + epsilon)

    return adv_image


def multi_restart_pgd_attack(model, input_image, label, epsilon, alpha, num_iterations, num_restarts=10):
    """多重启PGD：多次随机初始化，选最优攻击样本，攻击成功率拉满"""
    model.eval()
    device = input_image.device
    
    best_adv = input_image.clone()
    best_loss = -float('inf')
    
    # 多次重启RandPGD
    for restart in range(num_restarts):
        # 单次RandPGD攻击
        adv_image = input_image.clone()
        random_noise = torch.FloatTensor(adv_image.shape).uniform_(-epsilon, epsilon).to(device)
        adv_image = torch.clamp(adv_image + random_noise, input_image - epsilon, input_image + epsilon)
        adv_image = torch.clamp(adv_image, 0, 1)
        
        for _ in range(num_iterations):
            model.zero_grad()
            adv_image = adv_image.clone().detach().requires_grad_(True)
            output = model(adv_image)
            loss = nn.CrossEntropyLoss()(output, label)
            
            loss.backward()
            gradient = adv_image.grad.data.sign()
            adv_image = adv_image + alpha * gradient
            adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)
            adv_image = torch.clamp(adv_image, 0, 1)
        
        # 评估本次重启的攻击效果
        with torch.no_grad():
            final_output = model(adv_image)
            final_loss = nn.CrossEntropyLoss()(final_output, label).item()
        
        # 保存最优结果
        if final_loss > best_loss:
            best_loss = final_loss
            best_adv = adv_image.clone()
    
    return best_adv


def nes_pgd_attack(model, input_image, label, epsilon, alpha, num_iterations, momentum=0.9):
    """Nesterov加速的PGD（攻击强度远高于普通MIM-PGD）"""
    model.eval()
    device = input_image.device
    
    # 1. 随机初始化起始点（继承RandPGD的优势）
    adv_image = input_image.clone()
    random_noise = torch.FloatTensor(adv_image.shape).uniform_(-epsilon, epsilon).to(device)
    adv_image = torch.clamp(adv_image + random_noise, input_image - epsilon, input_image + epsilon)
    adv_image = torch.clamp(adv_image, 0, 1)
    
    grad_momentum = torch.zeros_like(adv_image).to(device)
    
    for _ in range(num_iterations):
        model.zero_grad()
        
        # 2. Nesterov核心：提前沿动量方向更新，计算"前瞻梯度"
        nes_adv = adv_image + momentum * alpha * grad_momentum.sign()  # 前瞻步
        nes_adv = nes_adv.clone().detach().requires_grad_(True)
        
        # 3. 计算前瞻点的梯度
        output = model(nes_adv)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        gradient = nes_adv.grad.data
        
        # 4. 动量更新（结合前瞻梯度）
        grad_momentum = momentum * grad_momentum + gradient / torch.norm(gradient, p=1, dim=[1,2,3], keepdim=True)
        gradient_sign = grad_momentum.sign()
        
        # 5. 更新并投影约束
        adv_image = adv_image + alpha * gradient_sign
        adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)
        adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image


def cw_pgd_hybrid_attack(model, input_image, label, epsilon, alpha, num_iterations, kappa=0.0):
    """CW-PGD混合攻击：结合CW损失和PGD迭代，突破鲁棒防御模型"""
    model.eval()
    device = input_image.device
    
    adv_image = input_image.clone()
    # 随机初始化
    random_noise = torch.FloatTensor(adv_image.shape).uniform_(-epsilon, epsilon).to(device)
    adv_image = torch.clamp(adv_image + random_noise, input_image - epsilon, input_image + epsilon)
    adv_image = torch.clamp(adv_image, 0, 1)
    
    for _ in range(num_iterations):
        model.zero_grad()
        adv_image = adv_image.clone().detach().requires_grad_(True)
        
        output = model(adv_image)
        # 1. CW损失函数（核心改进）
        # 获取真实标签的logit
        logit_true = output.gather(1, label.unsqueeze(1)).squeeze(1)
        # 获取除真实标签外最大的logit
        logit_other = torch.max(output - 1e4 * F.one_hot(label, num_classes=output.shape[1]), dim=1)[0]
        # CW损失：max(logit_other - logit_true + kappa, 0)（让其他标签logit超过真实标签）
        loss = torch.max(logit_other - logit_true + kappa, torch.zeros_like(logit_true)).mean()
        
        # 2. PGD迭代更新
        loss.backward()
        gradient = adv_image.grad.data.sign()
        adv_image = adv_image + alpha * gradient
        
        # 3. 投影约束
        adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)
        adv_image = torch.clamp(adv_image, 0, 1)
    
    return adv_image


def multi_model_pgd_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量
    label,               # 真实标签
    epsilon=0.5,         # PGD扰动上限
    alpha=0.05,          # 单次迭代步长
    num_iterations=50    # 迭代次数
):
    # 注意力机制参数（内部默认，不暴露接口）
    attention_ratio = 0.2  # 关注梯度最大的20%像素（可调整）
    
    valid_models = []
    valid_weights = []
    input_ori = input_image.clone().detach()
    
    # 1. 筛选分类正确的有效模型（保留原逻辑）
    for idx, model in enumerate(model_list):
        model.eval()
        with torch.no_grad():
            output = model(input_ori)
            pred_label = torch.argmax(output, dim=1)
            if pred_label == label:
                valid_models.append(model)
                valid_weights.append(model_weights[idx])
    
    if len(valid_models) == 0:
        print(f"警告：当前样本无分类正确的模型，返回原始图像")
        return input_image
    
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=input_image.device)
    valid_weights = valid_weights / valid_weights.sum()  # 权重归一化
    
    # 2. 预计算固定的注意力掩码（仅计算一次，核心修改）
    adv_image_init = input_image.clone().detach().requires_grad_(True)
    fusion_gradient_init = torch.zeros_like(adv_image_init)
    
    # 2.1 计算初始的多模型融合梯度
    for idx, model in enumerate(valid_models):
        model.zero_grad()
        output = model(adv_image_init)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        fusion_gradient_init += valid_weights[idx] * adv_image_init.grad.data.sign()
    
    # 2.2 生成固定的注意力掩码（仅基于初始梯度）
    grad_abs = torch.abs(fusion_gradient_init)
    grad_flat = grad_abs.view(grad_abs.shape[0], -1)
    k = int(grad_flat.shape[1] * attention_ratio)
    k = max(k, 1)  # 避免k=0
    threshold = torch.kthvalue(grad_flat, grad_flat.shape[1] - k, dim=1)[0]
    threshold = threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    attention_mask = (grad_abs >= threshold).float()  # 固定mask，后续不再修改
    attention_mask = attention_mask.detach()  # 解除梯度追踪
    
    # 3. PGD核心迭代（复用固定的注意力掩码）
    adv_image = input_image.clone()
    for _ in range(num_iterations):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        fusion_gradient = torch.zeros_like(adv_image)
        
        # 3.1 计算当前迭代的多模型融合梯度
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            output = model(adv_image)
            loss = nn.CrossEntropyLoss()(output, label)
            loss.backward()
            fusion_gradient += valid_weights[idx] * adv_image.grad.data.sign()
        
        # 3.2 复用固定的注意力掩码施加扰动（核心修改）
        perturbation = alpha * fusion_gradient.sign() * attention_mask
        
        # 3.3 投影约束（保留原逻辑）
        adv_image = adv_image + perturbation
        adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
        adv_image = adv_image.detach()
    
    return adv_image
    
# def multi_model_pgd_attack(
#     model_list,          # 参与攻击的模型列表
#     model_weights,       # 各模型的梯度权重（与model_list一一对应）
#     input_image,         # 输入图像张量
#     label,               # 真实标签
#     epsilon=0.5,         # PGD扰动上限
#     alpha=0.05,          # 单次迭代步长
#     num_iterations=50    # 迭代次数
# ):
#     valid_models = []
#     valid_weights = []
#     input_ori = input_image.clone().detach()
    
#     for idx, model in enumerate(model_list):
#         model.eval()
#         with torch.no_grad():
#             output = model(input_ori)
#             pred_label = torch.argmax(output, dim=1)
#             if pred_label == label:
#                 valid_models.append(model)
#                 valid_weights.append(model_weights[idx])
    
#     if len(valid_models) == 0:
#         print(f"警告：当前样本无分类正确的模型，返回原始图像")
#         return input_image
    
#     valid_weights = torch.tensor(valid_weights, dtype=torch.float32)
#     valid_weights = valid_weights / valid_weights.sum()
    
#     # PGD核心迭代（融合多模型梯度）
#     adv_image = input_image.clone()
#     for _ in range(num_iterations):
#         adv_image = adv_image.clone().detach().requires_grad_(True)
#         fusion_gradient = torch.zeros_like(adv_image)
        
#         for idx, model in enumerate(valid_models):
#             model.zero_grad()
#             output = model(adv_image)
#             loss = nn.CrossEntropyLoss()(output, label)
#             loss.backward()
            
#             fusion_gradient += valid_weights[idx] * adv_image.grad.data.sign()
        
#         adv_image = adv_image + alpha * fusion_gradient.sign()
#         adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)
#         adv_image = adv_image.detach()
    
#     return adv_image