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
            adv_image = adv_image.detach()
        
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
        adv_image = adv_image.detach()
    
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
        adv_image = adv_image.detach()
    
    return adv_image


# def multi_model_pgd_attack(
#     model_list,
#     model_weights,
#     input_image,
#     label,
#     epsilon=0.5,
#     alpha=0.05,
#     num_iterations=50
# ):
#     attention_ratio = 0.05  # 关注梯度最大的10%像素
#     momentum = 0.9         # MIM动量系数（经典值，可调整0.8-0.95）
    
#     valid_models = []
#     valid_weights = []
#     input_ori = input_image.clone().detach()
    
#     # 1. 筛选分类正确的有效模型（保留原逻辑）
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
    
#     valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=input_image.device)
#     valid_weights = valid_weights / valid_weights.sum()  # 权重归一化
    
#     # 2. 预计算固定的注意力掩码（仅计算一次，保留原逻辑）
#     adv_image_init = input_image.clone().detach().requires_grad_(True)
#     fusion_gradient_init = torch.zeros_like(adv_image_init)
    
#     # 2.1 计算初始的多模型融合梯度
#     for idx, model in enumerate(valid_models):
#         model.zero_grad()
#         output = model(adv_image_init)
#         loss = nn.CrossEntropyLoss()(output, label)
#         loss.backward()
#         fusion_gradient_init += valid_weights[idx] * adv_image_init.grad.data.sign()
    
#     # 2.2 生成固定的注意力掩码（仅基于初始梯度）
#     grad_abs = torch.abs(fusion_gradient_init)
#     grad_flat = grad_abs.view(grad_abs.shape[0], -1)
#     k = int(grad_flat.shape[1] * attention_ratio)
#     k = max(k, 1)  # 避免k=0
#     threshold = torch.kthvalue(grad_flat, grad_flat.shape[1] - k, dim=1)[0]
#     threshold = threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#     attention_mask = (grad_abs >= threshold).float()
#     attention_mask = attention_mask.detach()
    
#     # 3. PGD核心迭代（引入MIM动量机制 + 复用固定注意力掩码）
#     adv_image = input_image.clone()
#     grad_momentum = torch.zeros_like(adv_image).to(input_image.device)
    
#     for _ in range(num_iterations):
#         adv_image = adv_image.clone().detach().requires_grad_(True)
#         fusion_gradient = torch.zeros_like(adv_image)
        
#         # 3.1 计算当前迭代的多模型融合梯度（保留原逻辑）
#         for idx, model in enumerate(valid_models):
#             model.zero_grad()
#             output = model(adv_image)
            
#             logit_true = output.gather(1, label.unsqueeze(1)).squeeze(1)  # 正确标签logit
#             logit_other = torch.max(output - 1e4 * F.one_hot(label, num_classes=output.shape[1]), dim=1)[0]  # 最大错误标签logit
#             loss = torch.max(logit_other - logit_true + 0.1, torch.zeros_like(logit_true)).mean()  # CW损失（kappa=0.1）
#             loss.backward()

#             # loss = nn.CrossEntropyLoss()(output, label)
#             # loss.backward()
#             fusion_gradient += valid_weights[idx] * adv_image.grad.data  # 注意：此处不再提前取sign，保留幅值用于动量计算
        
#         # 3.2 MIM动量梯度更新（核心新增逻辑）
#         # 归一化当前梯度（避免梯度幅值波动影响动量）
#         grad_norm = torch.norm(fusion_gradient.view(fusion_gradient.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
#         grad_norm = torch.clamp(grad_norm, min=1e-12)  # 防止除零
#         normalized_grad = fusion_gradient / grad_norm
#         # 累积动量：历史动量 + 当前归一化梯度
#         grad_momentum = momentum * grad_momentum + normalized_grad
#         # 取动量梯度的符号（用于扰动更新）
#         momentum_grad_sign = grad_momentum.sign()
        
#         # 3.3 施加扰动：动量梯度 × 固定注意力掩码（核心修改）
#         perturbation = alpha * momentum_grad_sign * attention_mask
        
#         # 3.4 投影约束（保留原逻辑）
#         adv_image = adv_image + perturbation
#         adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
#         adv_image = adv_image.detach()
    
#     return adv_image


def multi_model_max_pgd_attack(
    model_list,
    model_weights,
    input_image,
    label,
    epsilon=0.8,          # CIFAR10最优默认：8/255≈0.0314（L∞扰动上限）
    alpha=0.08,          # CIFAR10最优默认：epsilon/10≈0.00314
    num_iterations=40,      # CIFAR10最优默认：40轮迭代（平衡效果与速度）
    attention_ratio=0.1,   # CIFAR10最优默认：15%像素（32×32图像更小，需稍大比例）
    momentum_start=0.5,     # 自适应动量起始值
    momentum_end=0.95,      # 自适应动量终止值
    gaussian_sigma=1.0,     # 高斯平滑核sigma（适配32×32）
    cw_kappa=0.1            # CW损失kappa值（CIFAR10最优）
):
    # -------------------------- 1. 适配CIFAR10的高斯平滑核生成 --------------------------
    device = input_image.device
    c = input_image.shape[1]  # CIFAR10为3通道
    kernel_size = 3           # 3×3核适配32×32图像（避免过度平滑）
    # 生成1D高斯核
    kernel_1d = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2+1)**2 / (2 * gaussian_sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    # 扩展为2D高斯核
    kernel_2d = kernel_1d.view(1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, kernel_size)
    # 适配多通道（分组卷积）
    gaussian_kernel = kernel_2d.repeat(c, 1, 1, 1).to(device)
    
    valid_models = []
    valid_weights = []
    input_ori = input_image.clone().detach()
    
    # 2. 筛选分类正确的有效模型（保留原逻辑）
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
    
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=device)
    valid_weights = valid_weights / valid_weights.sum()  # 权重归一化
    
    # 3. 预计算固定的注意力掩码（保留原逻辑）
    adv_image_init = input_image.clone().detach().requires_grad_(True)
    fusion_gradient_init = torch.zeros_like(adv_image_init)
    
    # 3.1 计算初始的多模型融合梯度
    for idx, model in enumerate(valid_models):
        model.zero_grad()
        output = model(adv_image_init)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        fusion_gradient_init += valid_weights[idx] * adv_image_init.grad.data.sign()
    
    # 3.2 生成固定的注意力掩码（仅基于初始梯度）
    grad_abs = torch.abs(fusion_gradient_init)
    grad_flat = grad_abs.view(grad_abs.shape[0], -1)
    k = int(grad_flat.shape[1] * attention_ratio)
    k = max(k, 1)  # 避免k=0
    threshold = torch.kthvalue(grad_flat, grad_flat.shape[1] - k, dim=1)[0]
    threshold = threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    attention_mask = (grad_abs >= threshold).float()
    attention_mask = attention_mask.detach()
    
    # 4. PGD核心迭代（新增：自适应动量 + 梯度平滑）
    adv_image = input_image.clone()
    grad_momentum = torch.zeros_like(adv_image).to(device)
    
    for i in range(num_iterations):
        # -------------------------- 新增：自适应动量计算 --------------------------
        momentum = momentum_start + (momentum_end - momentum_start) * min(i / (num_iterations/2), 1.0)
        
        adv_image = adv_image.clone().detach().requires_grad_(True)
        fusion_gradient = torch.zeros_like(adv_image)
        
        # 4.1 计算当前迭代的多模型融合梯度（保留CW损失逻辑）
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            output = model(adv_image)
            
            logit_true = output.gather(1, label.unsqueeze(1)).squeeze(1)  # 正确标签logit
            logit_other = torch.max(output - 1e4 * F.one_hot(label, num_classes=output.shape[1]), dim=1)[0]  # 最大错误标签logit
            loss = torch.max(logit_other - logit_true + cw_kappa, torch.zeros_like(logit_true)).mean()  # CW损失
            loss.backward()

            fusion_gradient += valid_weights[idx] * adv_image.grad.data  # 保留幅值用于动量计算
        
        # 4.2 MIM动量梯度更新（保留核心逻辑）
        grad_norm = torch.norm(fusion_gradient.view(fusion_gradient.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_norm = torch.clamp(grad_norm, min=1e-12)  # 防止除零
        normalized_grad = fusion_gradient / grad_norm
        grad_momentum = momentum * grad_momentum + normalized_grad
        
        # -------------------------- 新增：梯度平滑去噪 --------------------------
        # 对动量梯度做高斯平滑，消除高频噪声（适配32×32图像）
        # 使用分组卷积保证各通道独立平滑，padding=1保持尺寸不变
        grad_momentum_smoothed = F.conv2d(
            grad_momentum, 
            gaussian_kernel, 
            padding=kernel_size//2, 
            groups=c
        )
        momentum_grad_sign = grad_momentum_smoothed.sign()
        
        # 4.3 施加扰动：动量梯度 × 固定注意力掩码
        perturbation = alpha * momentum_grad_sign * attention_mask
        
        # 4.4 投影约束（保留原逻辑，适配CIFAR10归一化后范围）
        adv_image = adv_image + perturbation
        adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
        adv_image = adv_image.detach()
    
    return adv_image


def multi_model_nes_pgd_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量
    label,               # 真实标签
    epsilon,             # PGD扰动上限
    alpha,               # 单次迭代步长
    num_iterations,      # 迭代次数
    momentum=0.9,        # Nesterov动量系数（默认0.9）
    attention_ratio=0.1  # 关注梯度最大的10%像素（内部参数，可调整）
):
    """带固定注意力掩码的多模型NES-PGD攻击（聚焦核心区域+Nesterov加速+多模型融合）"""
    # 1. 基础初始化与设备对齐
    for model in model_list:
        model.eval()  # 所有模型设为评估模式
    device = input_image.device
    input_ori = input_image.clone().detach()  # 保存原始图像用于投影约束
    
    # 2. 筛选有效模型（仅保留对原始图像分类正确的模型）
    valid_models = []
    valid_weights = []
    for idx, model in enumerate(model_list):
        with torch.no_grad():
            output = model(input_ori)
            pred_label = torch.argmax(output, dim=1)
            if pred_label == label:  # 仅保留分类正确的模型（保证梯度有效）
                valid_models.append(model)
                valid_weights.append(model_weights[idx])
    
    # 无有效模型时返回原始图像
    if len(valid_models) == 0:
        print(f"警告：当前样本无分类正确的模型，返回原始图像")
        return input_image
    
    # 权重归一化（保证多模型梯度融合的权重和为1）
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=device)
    valid_weights = valid_weights / valid_weights.sum()
    
    # -------------------------- 新增：预计算固定注意力掩码 --------------------------
    # 2.1 计算原始图像的多模型融合初始梯度（用于生成mask）
    init_adv = input_ori.clone().detach().requires_grad_(True)
    init_fusion_grad = torch.zeros_like(init_adv)
    for idx, model in enumerate(valid_models):
        model.zero_grad()
        output = model(init_adv)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        init_fusion_grad += valid_weights[idx] * init_adv.grad.data.sign()
    
    # 2.2 生成固定注意力掩码（仅基于初始梯度，后续不再修改）
    grad_abs = torch.abs(init_fusion_grad)
    grad_flat = grad_abs.view(grad_abs.shape[0], -1)
    k = int(grad_flat.shape[1] * attention_ratio)
    k = max(k, 1)  # 避免k=0（无像素被选中）
    # 计算top-k梯度的阈值
    threshold = torch.kthvalue(grad_flat, grad_flat.shape[1] - k, dim=1)[0]
    threshold = threshold.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 扩展维度匹配图像
    attention_mask = (grad_abs >= threshold).float()  # 硬掩码：1=攻击区域，0=不攻击区域
    attention_mask = attention_mask.detach()  # 解除梯度追踪，固定mask
    # ------------------------------------------------------------------------------
    
    # 3. 随机初始化起始点（继承RandPGD优势，NES-PGD核心）
    adv_image = input_image.clone()
    random_noise = torch.FloatTensor(adv_image.shape).uniform_(-epsilon, epsilon).to(device)
    adv_image = torch.clamp(adv_image + random_noise, input_ori - epsilon, input_ori + epsilon)
    
    # 4. 初始化动量梯度（NES-PGD核心）
    grad_momentum = torch.zeros_like(adv_image).to(device)
    
    # 5. NES-PGD核心迭代（融合多模型梯度 + Nesterov加速 + mask约束）
    for _ in range(num_iterations):
        # 清零所有模型的梯度
        for model in valid_models:
            model.zero_grad()
        
        # 5.1 Nesterov核心：计算前瞻点（沿动量方向提前更新）
        nes_adv = adv_image + momentum * alpha * grad_momentum.sign()  # 前瞻步
        nes_adv = nes_adv.clone().detach().requires_grad_(True)  # 前瞻点开启梯度追踪
        
        # 5.2 融合多模型的前瞻梯度
        fusion_gradient = torch.zeros_like(nes_adv)
        for idx, model in enumerate(valid_models):
            # 单模型前瞻梯度计算
            output = model(nes_adv)
            loss = nn.CrossEntropyLoss()(output, label)
            loss.backward()
            # 按权重累加梯度（保留梯度幅值用于动量归一化）
            fusion_gradient += valid_weights[idx] * nes_adv.grad.data
        
        # 5.3 动量更新（NES-PGD核心：结合前瞻梯度）
        # L1归一化梯度，避免幅值波动影响动量稳定性
        grad_norm = torch.norm(fusion_gradient.view(fusion_gradient.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
        grad_norm = torch.clamp(grad_norm, min=1e-12)  # 防止除零
        normalized_grad = fusion_gradient / grad_norm
        # 累积动量（历史动量 + 当前归一化前瞻梯度）
        grad_momentum = momentum * grad_momentum + normalized_grad
        gradient_sign = grad_momentum.sign()
        
        # 5.4 施加扰动（核心修改：仅在mask区域施加扰动）
        perturbation = alpha * gradient_sign * attention_mask  # 扰动 × mask
        
        # 5.5 更新对抗样本并投影约束
        adv_image = adv_image + perturbation
        # 约束在epsilon球内（L∞范数）
        adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
        adv_image = adv_image.detach()
    
    return adv_image


def multi_model_pgd_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量
    label,               # 真实标签
    epsilon=0.5,         # PGD扰动上限
    alpha=0.05,          # 单次迭代步长
    num_iterations=50,    # 迭代次数
    attention_ratio=0.1  # 关注梯度最大的10%像素（内部参数，可调整）
):

    kappa = 0.0
    momentum=0.9

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
    momentum_buffer = torch.zeros_like(adv_image)

    for _ in range(num_iterations):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        fusion_gradient = torch.zeros_like(adv_image)
        
        # 3.1 计算当前迭代的多模型融合梯度
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            output = model(adv_image)
            
            # 1. CW损失函数（核心改进）
            logit_true = output.gather(1, label.unsqueeze(1)).squeeze(1)
            logit_other = torch.max(output - 1e4 * F.one_hot(label, num_classes=output.shape[1]), dim=1)[0]
            cw_loss = torch.max(logit_other - logit_true + kappa, torch.zeros_like(logit_true)).mean()
            ce_loss = nn.CrossEntropyLoss()(output, label)  # 交叉熵损失（分类错误时损失会上升）

            # 混合损失：CW损失为主，交叉熵损失为辅
            loss = cw_loss + 0.2 * ce_loss  # 0.2是交叉熵的权重，可调整
            loss.backward()
            fusion_gradient += valid_weights[idx] * adv_image.grad.data.sign()
        
        # ===== 关键修改2：更新动量缓存 =====
        # 1. 对当前融合梯度做L1归一化（避免幅值波动）
        grad_l1_norm = torch.norm(fusion_gradient, p=1, dim=[1,2,3], keepdim=True)
        grad_normalized = fusion_gradient / (grad_l1_norm + 1e-8)
        # 2. 动量更新：历史动量*衰减系数 + 归一化当前梯度
        momentum_buffer = momentum * momentum_buffer + grad_normalized
        
        # ===== 关键修改3：用动量计算扰动（替代原融合梯度）=====
        # 扰动方向由动量的符号决定，仍保留注意力掩码
        perturbation = alpha * momentum_buffer.sign() * attention_mask
        # perturbation = alpha * fusion_gradient.sign() * attention_mask
        
        # 3.3 投影约束
        adv_image = adv_image + perturbation
        adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
        adv_image = torch.clamp(adv_image, input_ori.min(), input_ori.max())
        adv_image = adv_image.detach()
    
    return adv_image


def multi_model_mask_cw_mi_pgd_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量
    label,               # 真实标签
    epsilon=0.5,         # PGD扰动上限
    alpha=0.05,          # 单次迭代步长
    num_iterations=50,   # 迭代次数
    attention_ratio=0.1,    # 关注梯度最大的10%像素（内部参数，可调整）
    kappa = 0.0,
    momentum=0.9  
):
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
    momentum_buffer = torch.zeros_like(adv_image)

    for _ in range(num_iterations):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        fusion_gradient = torch.zeros_like(adv_image)
        
        # 3.1 计算当前迭代的多模型融合梯度
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            output = model(adv_image)
            
            # 1. CW损失函数（核心改进）
            logit_true = output.gather(1, label.unsqueeze(1)).squeeze(1)
            logit_other = torch.max(output - 1e4 * F.one_hot(label, num_classes=output.shape[1]), dim=1)[0]
            cw_loss = torch.max(logit_other - logit_true + kappa, torch.zeros_like(logit_true)).mean()
            ce_loss = nn.CrossEntropyLoss()(output, label)  # 交叉熵损失（分类错误时损失会上升）

            # 混合损失：CW损失为主，交叉熵损失为辅
            loss = cw_loss + 0.2 * ce_loss  # 0.2是交叉熵的权重，可调整
            loss.backward()
            fusion_gradient += valid_weights[idx] * adv_image.grad.data.sign()
        
        # ===== 关键修改2：更新动量缓存 =====
        # 1. 对当前融合梯度做L1归一化（避免幅值波动）
        grad_l1_norm = torch.norm(fusion_gradient, p=1, dim=[1,2,3], keepdim=True)
        grad_normalized = fusion_gradient / (grad_l1_norm + 1e-8)
        momentum_buffer = momentum * momentum_buffer + grad_normalized
        perturbation = alpha * momentum_buffer.sign() * attention_mask
        
        # 3.3 投影约束
        adv_image = adv_image + perturbation
        adv_image = torch.clamp(adv_image, input_ori - epsilon, input_ori + epsilon)
        adv_image = torch.clamp(adv_image, input_ori.min(), input_ori.max())
        adv_image = adv_image.detach()
    
    return adv_image