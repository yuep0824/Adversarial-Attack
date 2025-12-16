import torch
import torch.nn as nn


def fgsm_attack(model, input_image, label, epsilon):
    model.eval()

    output = model(input_image)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()

    gradient = input_image.grad.data.sign()
    adv_image = torch.clamp(input_image + epsilon * gradient, input_image.min(), input_image.max())

    return adv_image


def multi_model_fgsm_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的梯度权重（与model_list一一对应）
    input_image,         # 输入图像张量 (shape: [1, C, H, W])
    label,               # 真实标签张量 (shape: [1])
    epsilon=0.4          # FGSM扰动强度
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
    
    valid_weights = torch.tensor(valid_weights, dtype=torch.float32, device=input_image.device)
    valid_weights = valid_weights / valid_weights.sum()
    
    fusion_gradient = torch.zeros_like(input_image)
    
    input_adv = input_image.clone().detach().requires_grad_(True)
    for idx, model in enumerate(valid_models):
        model.zero_grad()
        
        output = model(input_adv)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward(retain_graph=True)
        
        if input_adv.grad is not None:
            fusion_gradient += valid_weights[idx] * input_adv.grad.data.sign()
        
        input_adv.grad.zero_()
    
    with torch.no_grad():
        fusion_gradient_sign = fusion_gradient.sign()
        adv_image = input_image + epsilon * fusion_gradient_sign
        adv_image = torch.clamp(adv_image, input_image.min(), input_image.max())
    
    return adv_image