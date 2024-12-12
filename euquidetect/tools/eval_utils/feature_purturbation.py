import torch

class feature_purturbation:
    def __init__(self, epsilon=0.03, alpha=0.01, num_iter=40):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def __call__(self, features, target_features):
        adv_features = features.clone().detach().requires_grad_(True)
        for _ in range(self.num_iter):
            loss = torch.nn.functional.mse_loss(adv_features, target_features.detach())
            loss.backward()
            with torch.no_grad():
                adv_features = adv_features + self.alpha * adv_features.grad.sign()
                perturbation = torch.clamp(adv_features - features, min=-self.epsilon, max=self.epsilon)
                adv_features = torch.clamp(features + perturbation, min=0, max=1).detach().requires_grad_(True)
        return adv_features
