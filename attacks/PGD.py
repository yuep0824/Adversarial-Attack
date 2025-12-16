import torch
import torch.nn as nn


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



def multi_model_pgd_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量
    label,               # 真实标签
    epsilon=0.5,         # PGD扰动上限
    alpha=0.05,          # 单次迭代步长
    num_iterations=50    # 迭代次数
):
    valid_models = []
    valid_weights = []
    input_ori = input_image.clone().detach()
    
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
    
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32)
    valid_weights = valid_weights / valid_weights.sum()
    
    # PGD核心迭代（融合多模型梯度）
    adv_image = input_image.clone()
    for _ in range(num_iterations):
        adv_image = adv_image.clone().detach().requires_grad_(True)
        fusion_gradient = torch.zeros_like(adv_image)
        
        for idx, model in enumerate(valid_models):
            model.zero_grad()
            output = model(adv_image)
            loss = nn.CrossEntropyLoss()(output, label)
            loss.backward()
            
            fusion_gradient += valid_weights[idx] * adv_image.grad.data.sign()
        
        adv_image = adv_image + alpha * fusion_gradient.sign()
        adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)
        adv_image = adv_image.detach()
    
    return adv_image