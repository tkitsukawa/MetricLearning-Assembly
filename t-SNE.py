from torchsummary import summary
from estimate_image import *

from datasets import *
from models import *
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from datasets import *
from models import *
from losses import *
from train import *
from eval import *
from confusion_matrix import *
from models import *
from datasets import *
from losses import *
from estimate_image import *

def Transform_for_Model(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    transform = transforms.Compose([
        # 画像のサイズを変換する
        transforms.Resize((448, 448)),
        # # ランダムに明るさ、コントラスト、彩度、色相を変化させる
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        # # cropping
        # transforms.RandomCrop(size=(400, 400), padding=(5, 5)),
        # 画像のサイズを変換する
        # transforms.Resize((448, 448)),

        transforms.ToTensor(),
        # 指定された平均と標準偏差で画像を規格化する
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])

    image = transform(image)

    # 次元の追加，[チャンネル数，高さ，幅]から[バッチ数（１），チャンネル数，高さ，幅]
    image = image[np.newaxis, :, :, :]
    print(image.shape)
    return image


if __name__ == '__main__':
    # Datasetの準備
    train_dic = Make_Datapath_Dic("train")
    test_dic = Make_Datapath_Dic("test")
    imple_dic = Make_Datapath_Dic("implementation")

    print("train_dic:", train_dic)
    print("train_dic.type:", type(train_dic))
    print("train_dic.len:", len(train_dic))

    print("imple_dic:", imple_dic)
    print("imple_dic.type:", type(imple_dic))
    print("imple_dic.len:", len(imple_dic))

    transform = ImageTransform()

    test_dataset = TripletDataset(test_dic, transform=transform, phase="test")
    test_dataset_transform = TripletDataset(test_dic, transform=transform, phase="train")
    train_dataset = TripletDataset(train_dic, transform=transform, phase="test")
    train_dataset_transform = TripletDataset(train_dic, transform=transform, phase="train")
    imple_dataset = TripletDataset(imple_dic, transform=transform, phase="test")

    batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader_transform = DataLoader(test_dataset_transform, batch_size=batch_size, shuffle=False, num_workers=4)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_dataloader_transform = DataLoader(train_dataset_transform, batch_size=batch_size, shuffle=False, num_workers=4)
    imple_dataloader = DataLoader(imple_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("DatasetLoad__OK")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletNet().to(device)
    model.eval()
    model_dir = "output/210920_adaptive_Tripletloss/convertional_10000epc/"
    model_weights = model_dir + "model_all.pth"
    model.load_state_dict(torch.load(model_weights))

    summary(model, (3, 448, 448))

    # predicted_metrics = []
    # test_labels = []
    # with torch.no_grad():
    #     for i, (anchor, _, _, label, _) in enumerate(test_dataloader):
    #         anchor = anchor.to(device)
    #         metric = model(anchor).detach().cpu().numpy()
    #         print("metric_size:", metric.size)
    #         print("metric_type:", type(metric))
    #         print("metric_shape:", metric.shape)
    #         print("metric_reshape:", metric.shape)
    #         metric = metric.reshape(metric.shape[0], metric.shape[1])
    #         predicted_metrics.append(metric)
    #         test_labels.append(label.detach().numpy())
    #
    #     for k in range(9):
    #
    #         for i, (anchor, _, _, label, _) in enumerate(test_dataloader_transform):
    #             anchor = anchor.to(device)
    #             metric = model(anchor).detach().cpu().numpy()
    #             print("metric_size:", metric.size)
    #             print("metric_type:", type(metric))
    #             print("metric_shape:", metric.shape)
    #             print("metric_reshape:", metric.shape)
    #             metric = metric.reshape(metric.shape[0], metric.shape[1])
    #             predicted_metrics.append(metric)
    #             test_labels.append(label.detach().numpy())
    #
    # predicted_metrics_train = []
    # train_labels = []
    # with torch.no_grad():
    #     for i, (anchor, _, _, label, _) in enumerate(train_dataloader):
    #         anchor = anchor.to(device)
    #         metric = model(anchor).detach().cpu().numpy()
    #         print("metric_size:", metric.size)
    #         print("metric_type:", type(metric))
    #         print("metric_shape:", metric.shape)
    #         print("metric_reshape:", metric.shape)
    #         metric = metric.reshape(metric.shape[0], metric.shape[1])
    #         predicted_metrics_train.append(metric)
    #         train_labels.append(label.detach().numpy())
    #
    #
    # # print("predicted_metrics____:", predicted_metrics)
    # print("predicted_metrics____len:", len(predicted_metrics))
    # print("predicted_metrics____type:", type(predicted_metrics))
    # predicted_metrics_np = np.array(predicted_metrics)
    # predicted_metrics_train_np = np.array(predicted_metrics_train)
    # print("predicted_metrics____size:", predicted_metrics_np.size)
    # print("predicted_metrics____shape:", predicted_metrics_np.shape)
    # print("predicted_metrics_train____size:", predicted_metrics_train_np.size)
    # print("predicted_metrics_train____shape:", predicted_metrics_train_np.shape)
    #
    # predicted_metrics = np.concatenate(predicted_metrics, 0)
    # predicted_metrics_train = np.concatenate(predicted_metrics_train, 0)
    # test_labels = np.concatenate(test_labels, 0)
    # train_labels = np.concatenate(train_labels, 0)
    #
    # print("predicted_metrics:", predicted_metrics)
    # print("predicted_metrics.type:", type(predicted_metrics))
    # print("predicted_metrics.shape:", predicted_metrics.shape)
    # print("predicted_metrics_train.shape:", predicted_metrics_train.shape)
    # print("test_label:", test_labels)
    # print("test_label.type:", type(test_labels))
    # print("test_label:", test_labels.shape)
    #
    #
    #
    # test_labels_knn = test_labels.reshape([600, 1])
    # train_labels_knn = train_labels.reshape([40, 1])
    # # print("label_knn",  test_labels_knn)
    #
    # # 画像の読み込み
    # img_dir0 = "input/0.jpg"
    # img_dir1 = "input/1.jpg"
    # img_dir2 = "input/2.jpg"
    # frame_cutout0 = cv2.imread(img_dir0)
    # frame_cutout1 = cv2.imread(img_dir1)
    # frame_cutout2 = cv2.imread(img_dir2)
    # # transform for model
    # frame_cutout_model0 = Transform_for_Model(frame_cutout0)
    # frame_cutout_model1 = Transform_for_Model(frame_cutout1)
    # frame_cutout_model2 = Transform_for_Model(frame_cutout2)
    # # Progress Estimation
    # step_number0, error, img_embedding_np0 = Estimate_Image(model, frame_cutout_model0, predicted_metrics_train,
    #                                                         train_labels_knn)
    # step_number1, error, img_embedding_np1 = Estimate_Image(model, frame_cutout_model1, predicted_metrics_train,
    #                                                         train_labels_knn)
    # step_number2, error, img_embedding_np2 = Estimate_Image(model, frame_cutout_model2, predicted_metrics_train,
    #                                                         train_labels_knn)
    # print("Step:", step_number0)
    # print("Step:", step_number1)
    # print("Step:", step_number2)
    # predicted_metrics_train = np.append(predicted_metrics_train, img_embedding_np0, axis=0)
    # predicted_metrics_train = np.append(predicted_metrics_train, img_embedding_np1, axis=0)
    # predicted_metrics_train = np.append(predicted_metrics_train, img_embedding_np2, axis=0)
    # train_labels_knn = np.append(train_labels_knn, 10)
    # train_labels_knn = np.append(train_labels_knn, 10)
    # train_labels_knn = np.append(train_labels_knn, 10)
    # train_labels_knn = np.reshape(train_labels_knn, (43, 1))

    test_acc, true_label_list, pred_label_list = eval_Triplet(model, imple_dataloader, train_dataloader)

    print("test_acc:::::::::::::", test_acc)
    print("true_label_list::::::", true_label_list)
    print("pred_label_list::::::", pred_label_list)

    # tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(predicted_metrics)
    #
    # plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=test_labels, s=10)
    # # plt.xlim(-50, 50)
    # # plt.ylim(-50, 50)
    # plt.colorbar()
    # plt.savefig(model_dir + 'T-SNE4.png')
    # plt.show()
    #
    # tSNE_metrics = TSNE(n_components=2, random_state=0).fit_transform(predicted_metrics_train)
    #
    # plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=train_labels_knn, s=10)
    # # plt.xlim(-50, 50)
    # # plt.ylim(-50, 50)
    # plt.colorbar()
    # plt.savefig(model_dir + 'T-SNE4.png')
    # plt.show()
