import torch
import torch.nn as nn


def pc_i_fgsm_attack(model, original_images, labels, epsilon, num_iterations=10, num_predictions=1):
    criterion = nn.CrossEntropyLoss()

    alpha = epsilon / num_iterations
    perturbed_images = original_images.clone().detach().requires_grad_(True)
    ori_perturbed_images = original_images.clone().detach().requires_grad_(True)

    # 用于校正阶段
    original_outputs = model(ori_perturbed_images)
    original_loss = criterion(original_outputs, labels)
    model.zero_grad()
    original_loss.backward()
    original_gradient = ori_perturbed_images.grad.data

    for _ in range(num_iterations):
        # 预测阶段
        acumulated_predicted_gradients = torch.zeros_like(original_images).detach().to(original_images.device)
        # 先更新一次对抗样本
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images.detach().requires_grad_(True)
        for _ in range(num_predictions-1):
            outputs = model(perturbed_images)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_images.grad.data
            # 更新对抗样本（预测步骤）
            perturbed_images = perturbed_images + alpha * data_grad.sign()
            perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
            perturbed_images = perturbed_images.detach().requires_grad_(True)
            acumulated_predicted_gradients += data_grad / torch.sum(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)

        # 校正阶段
        corrected_gradient = original_gradient + acumulated_predicted_gradients
        # 更新对抗样本（校正步骤）
        perturbed_images = original_images + epsilon * corrected_gradient.sign()
        perturbed_images = torch.clamp(perturbed_images, original_images - epsilon, original_images + epsilon)
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    return perturbed_images


def multi_model_pc_i_fgsm_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量 (shape: [1, C, H, W])
    label,               # 真实标签张量 (shape: [1])
    epsilon=0.5,         # 最大扰动幅值
    num_iterations=10,   # 迭代次数
    num_predictions=1    # 预测阶段的预测次数
):
    """
    对齐multi_model_fgsm_attack风格的多模型PC-I-FGSM攻击
    核心逻辑：保留PC-I-FGSM的预测+校正阶段，适配单样本输入，筛选有效模型参与梯度融合
    """
    # ========== 步骤1：筛选有效模型（仅保留对当前样本分类正确的模型） ==========
    valid_models = []
    valid_weights = []
    input_ori = input_image.clone().detach()  # 原始图像备份
    
    for idx, model in enumerate(model_list):
        model.eval()  # 确保模型处于评估模式
        with torch.no_grad():
            output = model(input_ori)
            pred_label = torch.argmax(output, dim=1)
            if pred_label == label:  # 仅保留分类正确的模型
                valid_models.append(model)
                valid_weights.append(model_weights[idx])
    
    # 异常处理：无有效模型时返回原始图像
    if len(valid_models) == 0:
        print(f"警告：当前样本无分类正确的模型，返回原始图像")
        return input_image
    
    # 有效权重归一化（确保权重和为1）
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=input_image.device)
    valid_weights = valid_weights / valid_weights.sum()

    # ========== 步骤2：初始化变量（对齐参考代码命名风格） ==========
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon / num_iterations  # 每次迭代步长
    perturbed_images = input_image.clone().detach().requires_grad_(True)
    ori_perturbed_images = input_image.clone().detach().requires_grad_(True)

    # ========== 步骤3：校正阶段-计算原始梯度（多模型加权） ==========
    original_gradient = torch.zeros_like(input_image)
    for idx, model in enumerate(valid_models):
        model.zero_grad()
        # 计算单模型原始损失
        outputs = model(ori_perturbed_images)
        loss = criterion(outputs, label)
        loss.backward(retain_graph=True)  # 保留计算图，避免多模型梯度丢失
        
        if ori_perturbed_images.grad is not None:
            # 加权累加原始梯度（参考FGSM的梯度融合方式）
            original_gradient += valid_weights[idx] * ori_perturbed_images.grad.data
        ori_perturbed_images.grad.zero_()  # 清零当前模型梯度

    # ========== 步骤4：迭代生成对抗样本（预测+校正） ==========
    for _ in range(num_iterations):
        # 初始化预测阶段梯度累加器
        acumulated_predicted_gradients = torch.zeros_like(input_image)
        
        # ---------- 预测阶段 ----------
        # 先计算一次初始梯度
        pred_gradient = torch.zeros_like(input_image)
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            outputs = model(perturbed_images)
            loss = criterion(outputs, label)
            loss.backward(retain_graph=True)
            
            if perturbed_images.grad is not None:
                pred_gradient += valid_weights[idx] * perturbed_images.grad.data
            perturbed_images.grad.zero_()
        
        perturbed_images = perturbed_images.detach().requires_grad_(True)
        
        # 多次预测并更新对抗样本
        for _ in range(num_predictions - 1):
            curr_pred_gradient = torch.zeros_like(input_image)
            for idx, model in enumerate(valid_models):
                model.zero_grad()
                outputs = model(perturbed_images)
                loss = criterion(outputs, label)
                loss.backward(retain_graph=True)
                
                if perturbed_images.grad is not None:
                    curr_pred_gradient += valid_weights[idx] * perturbed_images.grad.data
                perturbed_images.grad.zero_()
            
            # 预测步骤更新对抗样本（参考FGSM的梯度符号更新方式）
            perturbed_images = perturbed_images + alpha * curr_pred_gradient.sign()
            # 限制扰动范围
            perturbed_images = torch.clamp(perturbed_images, input_ori.min(), input_ori.max())
            perturbed_images = perturbed_images.detach().requires_grad_(True)
            
            # 累加预测梯度（归一化后，避免梯度尺度失控）
            grad_norm = torch.sum(torch.abs(curr_pred_gradient), dim=(1, 2, 3), keepdim=True)
            grad_norm = torch.where(grad_norm == 0, torch.ones_like(grad_norm), grad_norm)
            acumulated_predicted_gradients += curr_pred_gradient / grad_norm

        # ---------- 校正阶段 ----------
        # 融合原始梯度和累加的预测梯度
        corrected_gradient = original_gradient + acumulated_predicted_gradients
        
        # 校正步骤更新对抗样本（参考FGSM的梯度符号+epsilon）
        with torch.no_grad():
            perturbed_images = input_ori + epsilon * corrected_gradient.sign()
            # 严格限制扰动范围（对齐参考代码的clamp逻辑）
            perturbed_images = torch.clamp(perturbed_images, input_ori.min(), input_ori.max())
        perturbed_images = perturbed_images.detach().requires_grad_(True)

    # ========== 步骤5：返回最终对抗样本（取消梯度追踪） ==========
    return perturbed_images.detach()