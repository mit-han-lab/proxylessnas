import torch
from torch import nn
import math

class PFLDLoss(nn.Module):
    def __init__(self, n_landmark):
        super(PFLDLoss, self).__init__()
        self.n_landmark = n_landmark

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle,
                landmarks, train_batchsize):
        
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1) + 1.0
        
        _attribute_gt = torch.ones((*attribute_gt.shape[:-1], attribute_gt.shape[-1]+1), device=attribute_gt.device)
        _attribute_gt[:,:-1] = attribute_gt
        attributes_w_n = _attribute_gt.float()
        weight_attribute = torch.sum(attributes_w_n, axis=1)
        # l2_distant = torch.sum(
        #     (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        l2_distant = wing_loss(landmark_gt, landmarks, N_LANDMARK=self.n_landmark)
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant), torch.mean(weight_angle)
        # return torch.mean(weight_angle * l2_distant), torch.mean(l2_distant), torch.mean(weight_angle)


def smoothL1(y_true, y_pred, beta=1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae > beta, mae - 0.5 * beta,
                                 0.5 * mae**2 / beta),
                     axis=-1)
    return torch.mean(loss)


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=98):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x,
                         w * torch.log(1.0 + absolute_x / epsilon),
                         absolute_x - c)
    loss = torch.sum(losses, axis=[1, 2])
    return loss


if __name__ == "__main__":
    criterion = PFLDLoss()
    landmark_gt = torch.randn(64, 106*2)
    landmark_pred = torch.randn(64, 106*2)
    # attribute_gt = (torch.randn(64, 6) > 0) * 1
    attribute_gt = torch.zeros(64, 6)
    attribute_gt[0] = torch.tensor([1,1,1,1,1,1])
    angle_gt = torch.randn(64, 3)
    angle_pred = torch.randn(64, 3)
    weighted_loss, loss, angle_loss = criterion(attribute_gt, landmark_gt, angle_gt, angle_pred, landmark_pred, 64)
    
    