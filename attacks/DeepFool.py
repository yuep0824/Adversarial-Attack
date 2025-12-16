import copy
import numpy as np
import torch
from torch.autograd import Variable


def deepfool_attack(model, input_image, label, num_classes=10, overshoot=0.5, max_iter=20):
    model.eval()
    output_origin = model(input_image).cpu().data.numpy().flatten()
    I = (np.array(output_origin)).flatten().argsort()[::-1]

    # 扰动初始化
    input_shape = input_image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(input_image)
    w = np.zeros(input_shape) # 扰动权重
    r_total = np.zeros(input_shape)   # 扰动总和

    
    test_image = Variable(pert_image, requires_grad=True)
    outputs = model(test_image)
    k_i = label
    loop_i = 0

    # 当k_i！=label时，即对抗样本越过了边界，结束循环
    while k_i == label and loop_i < max_iter:
        # 初始梯度
        pert = np.inf
        outputs[0, I[0]].backward(retain_graph=True)
        grad_origin = test_image.grad.data.cpu().numpy().copy()

        # 循环找最小的
        for k in range(1, num_classes):
            test_image.grad.zero_()
            # 第k个分类器，对pert_image的梯度，即wk
            outputs[0, I[k]].backward(retain_graph=True)
            current_grad = test_image.grad.data.cpu().numpy()

            w_k = current_grad - grad_origin         # wk，梯度差
            f_k = (outputs[0, I[k]] - outputs[0, I[0]]).cpu().data.numpy()  # fk，预测概率差
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())  # l'，距离度量

            # 取最短距离
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i =  (pert + 1e-4) * w / np.linalg.norm(w)
        r_total = np.float32(r_total + r_i)
        pert_image = input_image + (1 + overshoot) * torch.from_numpy(r_total)
        pert_image = torch.clamp(pert_image, input_image.min(), input_image.max())
        
        test_image = Variable(pert_image, requires_grad=True)
        outputs = model(test_image)
        k_i = np.argmax(outputs.data.cpu().numpy().flatten())

        loop_i += 1

    r_total = (1 + overshoot) * r_total

    return pert_image, loop_i


def multi_model_deepfool_attack(
    model_list,          # 参与攻击的模型列表
    model_weights,       # 各模型的权重（与model_list一一对应）
    input_image,         # 输入图像张量 (shape: [1, C, H, W])
    label,               # 真实标签张量 (shape: [1])
    num_classes=10,      # 分类类别数
    overshoot=0.5,       # 扰动超调量（增强攻击效果）
    max_iter=20          # 最大迭代次数
):
    device = input_image.device
    label_np = label.cpu().numpy().item()
    
    valid_models = []
    valid_weights = []
    input_ori = input_image.clone().detach()
    
    for idx, model in enumerate(model_list):
        model.eval()
        with torch.no_grad():
            output = model(input_ori)
            pred_label = torch.argmax(output, dim=1).cpu().numpy().item()
            if pred_label == label_np:
                valid_models.append(model)
                valid_weights.append(model_weights[idx])
    
    if len(valid_models) == 0:
        print(f"警告：当前样本无分类正确的模型，返回原始图像")
        return input_image, 0
    
    valid_weights = np.array(valid_weights, dtype=np.float32)
    valid_weights = valid_weights / valid_weights.sum()
    
    # 初始化扰动相关变量（和单模型DeepFool一致）
    input_shape = input_image.cpu().detach().numpy().shape
    r_total = np.zeros(input_shape, dtype=np.float32)
    pert_image = copy.deepcopy(input_image)
    loop_i = 0
    
    # 核心迭代：直到所有有效模型分类错误 或 达到最大迭代次数
    while loop_i < max_iter:
        all_wrong = True
        with torch.no_grad():
            for model in valid_models:
                output = model(pert_image)
                pred_label = torch.argmax(output, dim=1).cpu().numpy().item()
                if pred_label == label_np:
                    all_wrong = False
                    break
        if all_wrong:
            break
        
        # 初始化融合扰动（每次迭代重置）
        fusion_r_i = np.zeros(input_shape, dtype=np.float32)
        
        # 遍历每个有效模型，计算其DeepFool最小扰动并加权融合
        for model_idx, model in enumerate(valid_models):
            model.eval()
            # 复制当前对抗样本，开启梯度追踪
            test_image = pert_image.clone().detach().requires_grad_(True).to(device)
            outputs = model(test_image)
            
            output_np = outputs.cpu().data.numpy().flatten()
            I = np.argsort(output_np)[::-1]
            
            pert = np.inf
            w = np.zeros(input_shape)
            
            outputs[0, I[0]].backward(retain_graph=True)
            grad_origin = test_image.grad.data.cpu().numpy().copy()
            
            # 遍历其他类别，找最小扰动
            for k in range(1, num_classes):
                test_image.grad.zero_()
                outputs[0, I[k]].backward(retain_graph=True)
                current_grad = test_image.grad.data.cpu().numpy().copy()
                
                w_k = current_grad - grad_origin
                f_k = (outputs[0, I[k]] - outputs[0, I[0]]).cpu().data.numpy()
                
                w_k_flat = w_k.flatten()
                pert_k = abs(f_k) / (np.linalg.norm(w_k_flat) + 1e-8)
                
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            
            r_i = (pert + 1e-4) * w / (np.linalg.norm(w.flatten()) + 1e-8)
            fusion_r_i += valid_weights[model_idx] * r_i
        
        r_total = r_total + fusion_r_i
        pert_image = input_image + (1 + overshoot) * torch.from_numpy(r_total).to(device)
        pert_image = torch.clamp(pert_image, input_image.min(), input_image.max())
        
        loop_i += 1
    
    r_total = (1 + overshoot) * r_total
    return pert_image, loop_i
