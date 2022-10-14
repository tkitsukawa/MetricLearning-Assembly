import cv2
import numpy as np
import torch
from datasets import *
from losses import *
from models import *



def kNN(basis_data, basis_label, test_data):
    # setting k
    k = 4

    basis_label = np.array(basis_label).astype(np.float32)
    basis_data = np.array(basis_data).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(basis_data, cv2.ml.ROW_SAMPLE, basis_label)
    ret, test_label, neighbours, distance = knn.findNearest(test_data, k)

    # 近傍k点との距離の平均を計算
    distance = distance[0]
    distance_sum = 0
    for i in range(k):
        distance_sum = distance_sum + distance[i]

    distance_ave = distance_sum / k

    test_label = test_label[0]
    test_label = int(test_label[0])

    return test_label, distance_ave



def Basis_Embedding(model, train_kNN_loader):
    model.eval()

    dimension = 128
    data_length = 40
    kNN_basis_label_list = []

    # ０で埋められたNumpy配列を作成
    kNN_basis_np = np.zeros(shape=(data_length, dimension))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_idx, (anchor, _, _, anchor_label, _) in enumerate(train_kNN_loader, start=1):

        # 画像データをGPUに送る
        anchor = anchor.to(device)
        # データをGPUにあるモデルに入れる
        anc_embedding = model(anchor)
        # torch.tensorからnumpyに変換
        anc_embedding = anc_embedding.to('cpu').detach().numpy().copy()
        anchor_label = anchor_label.to('cpu').detach().numpy().copy()

        for i in range(len(anchor_label)):
            kNN_basis_label_list.append(anchor_label[i])
            number = i + (batch_idx - 1) * len(anchor_label)
            # print("number", number)
            kNN_basis_np[number] = anc_embedding[i]

    kNN_basis_label_np = np.array(kNN_basis_label_list)
    # print(kNN_basis_label_np)

    kNN_basis_label_np = np.reshape(kNN_basis_label_np, (data_length, 1))

    return kNN_basis_np, kNN_basis_label_np




def Estimate_Image(model, image, kNN_basis_np, kNN_basis_label_np):
    dimension = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 画像データをGPUに送る
    image = image.to(device)
    # データをGPUにあるモデルに入れる
    img_embedding = model(image)
    # torch.tensorからnumpyに変換
    img_embedding_np = img_embedding.to('cpu').detach().numpy().copy()
    img_embedding_np = np.reshape(img_embedding_np, (1, dimension))
    # kNNの関数に代入する
    pred_label, distance = kNN(kNN_basis_np, kNN_basis_label_np, img_embedding_np)

    print(distance)



    return pred_label, distance, img_embedding_np
