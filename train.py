import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import os
from torchsummary import summary

import matplotlib.pyplot as plt

from datasets import *
from losses import *
from models import *
from parameters import args

import time


def Train_Triplet(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.
    dis_pos_sum = 0.
    dis_neg_sum = 0.
    count = 0
    for batch_idx, (anchor, positive, negative, anchor_label, negative_label) in enumerate(train_loader, start=1):

        # print(type(anchor))
        # print(anchor.size())
        # print("anchor_label", anchor_label)
        # print("negative_label", negative_label)
        anchor_img = np.transpose(anchor[0], [1, 2, 0])
        anchor_img = anchor_img * 255
        anchor_img = np.array(anchor_img, dtype=np.uint8)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_RGBA2BGR)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)

        positive_img = np.transpose(positive[0], [1, 2, 0])
        positive_img = positive_img * 255
        positive_img = np.array(positive_img, dtype=np.uint8)
        positive_img = cv2.cvtColor(positive_img, cv2.COLOR_RGBA2BGR)
        positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2GRAY)

        negative_img = np.transpose(negative[0], [1, 2, 0])
        negative_img = negative_img * 255
        negative_img = np.array(negative_img, dtype=np.uint8)
        negative_img = cv2.cvtColor(negative_img, cv2.COLOR_RGBA2BGR)
        negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2GRAY)

        connected_image = cv2.hconcat([anchor_img, positive_img, negative_img])
        cv2.imshow("input_Image (Anchor_Image, Positive_Image, Negative_Image)", connected_image)
        cv2.waitKey(5)

        # 勾配を初期化
        optimizer.zero_grad()

        # 画像データをGPUに送る
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # print("anchor", type(anchor))
        # print("anchor.shape", anchor.shape)

        # データをGPUにあるモデルに入れる
        anc_embedding = model(anchor)
        pos_embedding = model(positive)
        neg_embedding = model(negative)

        loss_sum_batch = 0.
        distance_pos_sum_batch = 0.
        distance_neg_sum_batch = 0.

        for i in range(len(anchor_label)):
            # Lossを計算
            loss_one, distance_pos_one, distance_neg_one = criterion(
                anc_embedding[i],
                pos_embedding[i],
                neg_embedding[i],
                anchor_label[i],
                negative_label[i]
            )
            loss_sum_batch += loss_one
            distance_pos_sum_batch += distance_pos_one
            distance_neg_sum_batch += distance_neg_one
            # print("loss", loss_one)
            # print("loss.type", type(loss_one))

        loss = loss_sum_batch / len(anchor_label)
        distance_pos_per_batch = distance_pos_sum_batch / len(anchor_label)
        dis_pos_sum += distance_pos_per_batch
        distance_neg_per_batch = distance_neg_sum_batch / len(anchor_label)
        dis_neg_sum += distance_neg_per_batch
        # print(f"loss_sum:{loss_sum_batch}, len(anchor_label):{len(anchor_label)}, loss:{loss}")
        # 勾配を計算
        loss.backward()
        # optimizerで重みパラメータを更新
        optimizer.step()

        # batchごとにロスを足していく
        running_loss += loss.item()
        count += 1
        # print("Batch count:", count)

    train_loss = running_loss / count
    distance_pos = dis_pos_sum / count
    distance_neg = dis_neg_sum / count

    return train_loss, float(distance_pos), float(distance_neg)

