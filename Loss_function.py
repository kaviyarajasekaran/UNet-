import torch as t
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alfa=0.75, gamma=2):
        super().__init__()
        self.alfa = alfa
        self.gamma = gamma
        
    def forward(self, y_pred, y_truth):
        y_truth = y_truth.float()
        pt = torch.where(y_truth == 1, y_pred, 1 - y_pred)
        alfa = torch.where(y_truth == 1, self.alfa, 1 - self.alfa)
        pt = torch.clamp(pt, min=1e-5)
        loss = -(alfa * (1 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()

if __name__ == "__main__":
    y_truth = t.tensor([[0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 0]], dtype=t.float32)
    y_pred = t.tensor([[0.1, 0.2, 0.1],
                       [0.14, 0.1, 0.4],
                       [0.33, 0.5, 0.8]], dtype=t.float32)
    obj = FocalLoss()
    loss_value = obj(y_truth, y_pred)
    print("Focal Loss:", loss_value.item())
