import torch
import torch.nn as nn
import torch.optim as optim
import warnings


class CW():
    def __init__(self, model, targeted=False, c=1e-4, kappa=0, steps=100, lr=0.01):
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)

    # 这个函数用所有非目标标签的 logit 中最大值减去目标标签的 logit
    def f(self, x):
        outputs = self.model(x)
        one_hot_labels = torch.eye(outputs.size(1)).to(self.device)[self.labels]
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        if self.targeted:
            return torch.clamp(i - j, min=-self.kappa)
        else:
            return torch.clamp(j - i, min=-self.kappa)

    def attack(self, images, labels):
        self.labels = labels
        self.forward(images, labels)

        w = images.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([w], lr=self.lr)

        prev = float('inf')
        for step in range(self.steps):
            loss1 = torch.mean(torch.square(w - images))
            loss2 = torch.max(torch.tensor(0.0).to(self.device), self.f(w) - self.kappa)
 
            cost = loss1 + self.c * loss2
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if step % (self.steps // 10) == 0:
                if cost > prev:
                    warnings.warn("Early stopped cause the loss did not converge.")
                    return (1/2 * (nn.Tanh()(w) + 1)).detach()
                prev = cost

        return (1/2 * (nn.Tanh()(w) + 1)).detach()
