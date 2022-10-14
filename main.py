import os
import time
import csv
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from datasets import *
from models import *
from losses import *
from train import *
from eval import *
from test import *
from parameters import args
from confusion_matrix import *


def main():
    u = time.gmtime()
    print("year:", u.tm_year)
    print("month:", u.tm_mon)
    print("day:", u.tm_mday)
    print("hour", u.tm_hour + 9)
    print("minute:", u.tm_min)
    t_start = time.time()

    # Datasetの準備
    train_dic = Make_Datapath_Dic("train")
    test_dic = Make_Datapath_Dic("test")

    label_list = list(train_dic.keys())

    print("label_list", label_list)
    print("train_dic:", train_dic)
    print("train_dic.type:", type(train_dic))
    print("train_dic.len:", len(train_dic))

    transform = ImageTransform()

    train_dataset = TripletDataset(train_dic, transform=transform, phase="train")
    test_dataset = TripletDataset(test_dic, transform=transform, phase="test")
    train_kNN_dataset = TripletDataset(train_dic, transform=transform, phase="test")

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_kNN_dataloader = DataLoader(train_kNN_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    print("DatasetLoad__OK")

    # 学習に関する準備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletNet().to(device)
    # model = TripletResNet().to(device)
    # 損失関数の設定
    # criterion = TripletLoss()
    criterion = Adaptive_TripletLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[args.epochs//2, (args.epochs//4)*3], gamma=0.1
    # )

    summary(model, (3, 448, 448))
    torch.autograd.set_detect_anomaly(True)

    epoch_data = []
    train_loss_data = []
    test_acc_data = []

    epoch_data_X = []
    train_loss_data_X = []
    test_acc_data_X = []
    sum_tarain_loss = 0.
    sum_test_acc = 0.
    sum_true_label_list = []
    sum_pred_label_list = []

    sum_distance_pos = 0.
    sum_distance_neg = 0.
    distance_pos_data_X = []
    distance_neg_data_X = []


    # 保存関係
    train_date = str(u.tm_year) + "_" + str(u.tm_mon) +  "_" +  str(u.tm_mday) + "_" +  str(u.tm_hour + 9)
    print("folder_name:" + train_date)
    out_dir = "output//" + train_date
    os.makedirs(out_dir, exist_ok=True)

    print("TRAIN_START!!")

    # 実際の学習スタート，エポック数分回す
    for epoch in range(1, args.epochs+1):
        train_loss, distance_pos, distance_neg = Train_Triplet(model, train_dataloader, criterion, optimizer)
        test_acc, true_label_list, pred_label_list = eval_Triplet(model, test_dataloader, train_kNN_dataloader)

        # print(f"[{epoch}epoch] Positive Distance:{round(distance_pos, 3)}, Negative Distance:{round(distance_neg, 3)}")

        epoch_data.append(epoch)
        train_loss_data.append(train_loss)
        test_acc_data.append(test_acc)

        sum_tarain_loss += train_loss
        sum_test_acc += test_acc
        sum_distance_pos += distance_pos
        sum_distance_neg += distance_neg
        sum_true_label_list.extend(true_label_list)
        sum_pred_label_list.extend(pred_label_list)

        # XエポックごとにLossとAccuracyをグラフに表示
        X = 100
        if epoch % X == 0:
            train_loss_X = sum_tarain_loss / X
            test_acc_X = sum_test_acc / X
            distance_pos_X = sum_distance_pos / X
            distance_neg_X = sum_distance_neg / X

            epoch_data_X.append(epoch)
            train_loss_data_X.append(train_loss_X)
            test_acc_data_X.append(test_acc_X)
            distance_pos_data_X.append(distance_pos_X)
            distance_neg_data_X.append(distance_neg_X)


            sum_tarain_loss = 0.
            sum_test_acc = 0.
            sum_distance_pos = 0.
            sum_distance_neg = 0.

            print(f"[{epoch}epoch] Train Loss: {round(train_loss_X, 3)}, Test Accuracy: {round(test_acc_X, 3)}")
            # print(f"Positive Distance: {round(distance_pos_X, 3)}, Negative Distance: {round(distance_neg_X, 3)}")

        # Yエポックごとにやること
        if epoch % 5000 == 0:
            # Yエポックごとに混合行列の計算
            conf_matrix = confusion_matrix(sum_true_label_list, sum_pred_label_list, labels=label_list)
            # confusion matrixのプロット、保存、表示
            plot_confusion_matrix(conf_matrix, classes=label_list,
                                  output_file=out_dir + ("//%s_row.png" % epoch), normalize=False)
            plot_confusion_matrix(conf_matrix, classes=label_list, output_file=out_dir + ("//%s_normalize.png" % epoch),
                                  normalize=True)

            sum_true_label_list = []
            sum_pred_label_list = []

            # Yエポックごとにパラメータ調整されたモデルの保存
            # model = model.to('cpu')
            # model_name = out_dir + ("//%s_epoch.pth" % epoch)
            # torch.save(model.state_dict(), model_name)
            # print(f'Saved model as {model_name}')
            model = model.to(device)

    t_end = time.time()
    train_time = (t_end - t_start) / 3600



    # 学習終了，時間を表示
    print("--FiNISH-- Train Time(hour):", round(train_time, 1))

    # 全体の学習済みモデルの保存
    model = model.to('cpu')
    model_name = out_dir + "//model_all.pth"
    torch.save(model.state_dict(), model_name)
    print(f'Saved model as {model_name}')


    # Loss,Accuracyの結果をエクセルファイルとして保存
    with open(out_dir + "//data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_data)
        writer.writerow(train_loss_data)
        writer.writerow(test_acc_data)
        writer.writerow(epoch_data_X)
        writer.writerow(train_loss_data_X)
        writer.writerow(test_acc_data_X)
        writer.writerow(distance_pos_data_X)
        writer.writerow(distance_neg_data_X)

        Hyperparameter = ["batch size:", batch_size, "epoch:", args.epochs, "learning rate:", args.lr,
                          "train time(hour):", train_time]
        writer.writerow(Hyperparameter)

    # Lossのグラフの表示
    plt.figure()
    plt.plot(epoch_data_X, train_loss_data_X, color='blue', label='train loss')
    # plt.plot(x_epoch_data, y_test_loss_data, color='orange', label='Test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, )
    plt.grid()
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig(out_dir + "//Loss.png")

    # # Accuracyのグラフの表示
    plt.figure()
    # plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='Train')
    plt.plot(epoch_data_X, test_acc_data_X, color='orange', label='test accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(out_dir + '//Accuracy.png')

    # Distance表示
    plt.figure()
    plt.plot(epoch_data_X, distance_pos_data_X, color='green', label='Positive Distance')
    plt.plot(epoch_data_X, distance_neg_data_X, color='blue', label='Negative Distance')
    plt.title('Distance in embedding space')
    plt.xlabel('epoch')
    plt.ylabel('distance')
    plt.ylim(0, )
    plt.grid()
    plt.legend(loc="upper left")
    plt.savefig(out_dir + '//Distance.png')

    plt.show()


if __name__ == '__main__':
    main()
