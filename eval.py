import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import os
from torchsummary import summary
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from datasets import *
from losses import *
from models import *
from parameters import args


def kNN(basis_data, basis_label, test_data):
    # print("^^^^^^^^ basis_data", basis_data)
    # print("^^^^^^^^ basis_data", type(basis_data))
    # print("^^^^^^^^ basis_data", basis_data.shape)
    # print("^^^^^^^^ basis_label", basis_label)
    # print("^^^^^^^^ basis_label", type(basis_label))
    # print("^^^^^^^^ basis_label", basis_label.shape)
    # print("^^^^^^^^ testdata", test_data)
    # print("^^^^^^^^ testdata", type(test_data))
    # print("^^^^^^^^ testdata", test_data.shape)

    basis_label = np.array(basis_label).astype(np.float32)
    basis_data = np.array(basis_data).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(basis_data, cv2.ml.ROW_SAMPLE, basis_label)
    ret, test_label, neighbours, distance = knn.findNearest(test_data, 5)

    # print("result:\t{}".format(test_label))
    # print("neighbours:\t{}".format(neighbours))
    # print("distance:\t{}".format(distance))

    test_label = test_label[0]

    return test_label



def eval_Triplet(model, test_loader, train_kNN_loader):
    model.eval()

    dimension = 128
    data_length = 24
    kNN_basis_label_list = []

    # ０で埋められたNumpy配列を作成
    kNN_basis_np = np.zeros(shape=(data_length, dimension))

    sum_accuracy = 0.
    count = 0

    # 混同行列用list
    pred_label_list = []
    true_label_list = []


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (anchor, _, _, anchor_label, _) in enumerate(train_kNN_loader, start=1):

        # 画像データをGPUに送る
        anchor = anchor.to(device)
        # データをGPUにあるモデルに入れる
        anc_embedding = model(anchor)
        # torch.tensorからnumpyに変換
        anc_embedding = anc_embedding.to('cpu').detach().numpy().copy()
        anchor_label = anchor_label.to('cpu').detach().numpy().copy()

        # print(anc_embedding)
        # print(anc_embedding.shape)
        # print(anc_embedding[1])
        # print("---------", len(anc_embedding))
        # print(batch_idx)

        for i in range(len(anchor_label)):
            kNN_basis_label_list.append(anchor_label[i])
            number = i + (batch_idx - 1) * len(anchor_label)
            # print("number", number)
            kNN_basis_np[number] = anc_embedding[i]

    # print("-------kNN_basis_label_list", kNN_basis_label_list)
    # print("-------kNN_basis_np", kNN_basis_np)
    # print("-------kNN_basis_np.shape", kNN_basis_np.shape)

    kNN_basis_label_np = np.array(kNN_basis_label_list)

    # kNN_basis_np = np.reshape(kNN_basis_np, (data_length, 128))
    kNN_basis_label_np = np.reshape(kNN_basis_label_np, (data_length, 1))

    # print("kNN_basis_np:", kNN_basis_np)
    # print("kNN_basis_np.shape:", kNN_basis_np.shape)
    # print("kNN_basis_label_np:", kNN_basis_label_np)
    # print("kNN_basis_label_np.shape:", kNN_basis_label_np.shape)

    for batch_idx, (anchor, _, _, anchor_label, _) in enumerate(test_loader, start=1):

        # 画像データをGPUに送る
        anchor = anchor.to(device)
        # データをGPUにあるモデルに入れる
        anc_embedding = model(anchor)
        # torch.tensorからnumpyに変換
        anc_embedding_np = anc_embedding.to('cpu').detach().numpy().copy()
        anchor_label = anchor_label.to('cpu').detach().numpy().copy()

        # print("len(anchor_label):", len(anchor_label))

        for i in range(len(anchor_label)):
            count += 1

            # print("------------anc_embedding.shape", anc_embedding.shape)
            # print("------------anc_embedding[0]", anc_embedding[1])
            # print("------------anchor_label[0]", anchor_label[1])

            anc_embedding_np_one = np.reshape(anc_embedding_np[i], (1, dimension))

            # kNNの関数に代入する
            test_label = kNN(kNN_basis_np, kNN_basis_label_np, anc_embedding_np_one)
            # print(f"true: {anchor_label[i]}, predict: {int(test_label[0])}")
            true_label_list.append(anchor_label[i])
            pred_label_list.append(int(test_label[0]))

            if anchor_label[i] == test_label[0]:
                sum_accuracy += 1
                # print("------success-------")

    test_acc = sum_accuracy / count
    # print(f"sum_accuracy:{sum_accuracy}, count:{count}, test_accuracy:{test_acc}")

    conf_matrix = confusion_matrix(true_label_list, pred_label_list)

    # print(conf_matrix)

    return test_acc, true_label_list, pred_label_list






