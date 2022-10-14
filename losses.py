import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import norm


class TripletLoss(nn.Module):
    def __init__(self, margin=100.0):
        super().__init__()
        self.margin = margin

    # ユークリッド距離を計算
    def euclidean_distance(self, x1, x2):
        distance = (x1 - x2).pow(2).sum()
        if not distance == 0:
            distance = torch.sqrt(distance)
        return distance

    # , anchor_label, positive_label
    def forward(self, anchor, positive, negative, anchor_label, negative_label):
        distance_positive = self.euclidean_distance(anchor, positive)
        distance_negative = self.euclidean_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses, distance_positive, distance_negative



class Adaptive_TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # ユークリッド距離を計算
    def euclidean_distance(self, x1, x2):
        distance = (x1 - x2).pow(2).sum()
        if not distance == 0:
            distance = torch.sqrt(distance)
        return distance

    def forward(self, anchor, positive, negative, anchor_label, negative_label):

        margin = norm.pdf(negative_label, anchor_label, 3) * 5000

        distance_positive = self.euclidean_distance(anchor, positive)
        distance_negative = self.euclidean_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + margin)

        return losses, distance_positive, distance_negative


