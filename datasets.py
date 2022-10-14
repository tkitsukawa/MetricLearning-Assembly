import os
import glob
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image



# 名前:[データパスのリスト]をstepの数（今回は10）要素持つ辞書型のリストを作成
def Make_Datapath_Dic(phase='train'):
    # データセットのあるパスを以下三つから一つ選択
    # root_path = './datasets/' + phase + "/datasets"   # umelab_data
    # root_path = './datasets/trans/' + phase             # factory_class8
    root_path = './datasets/trans_class123456/' + phase        # factory_class5
    class_list = os.listdir(root_path)
    print("class_list::::::", class_list)
    class_list = [class_name for class_name in class_list if not class_name.startswith('.')]
    print("class_list:::::::::::", class_list)
    class_list = sorted(class_list)
    print("class_list:::::::::::::::", class_list)
    datapath_dic = {}
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name, '*.jpg')
        for path in glob.glob(target_path):
            data_list.append(path)
        datapath_dic[i] = data_list

    return datapath_dic



class ImageTransform():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([

                # 画像のサイズを変換する
                transforms.Resize((448, 448)),

                # # 90度回転させる（各向き4分の1の確率）
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.75),
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.66),
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.5),

                # ランダムに画像を切り抜く
                transforms.RandomApply(
                    [transforms.RandomCrop(size=(435, 435), padding=(10, 10))], p=7.0),

                # ランダムに明るさ、コントラスト、彩度、色相を変化させる
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.7),
                # ランダムに射影変換を行う
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.2)], p=0.7),
                # # ランダムに回転を行う
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=10)], p=0.7),

                # ランダムに画像を切り抜く
                transforms.RandomApply(
                    [transforms.RandomCrop(size=(435, 435), padding=(10, 10))], p=7.0),

                # 画像のサイズを変換する
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                # 指定された平均と標準偏差で画像を規格化する
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),

            'test': transforms.Compose([
                # 画像のサイズを変換する
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


class ImageTransform_v2():
    def __init__(self):
        self.data_transform = {
            'train': transforms.Compose([

                # 画像のサイズを変換する
                transforms.Resize((448, 448)),

                # 90度回転させる（各向き4分の1の確率）
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.75),
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.66),
                # transforms.RandomApply(
                #     [transforms.RandomRotation(degrees=(90, 90))], p=0.5),

                # ランダムに画像を切り抜く
                transforms.RandomApply(
                    [transforms.RandomCrop(size=(435, 435), padding=(20, 20))], p=7.0),

                # ランダムに明るさ、コントラスト、彩度、色相を変化させる
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.7),
                # ランダムに射影変換を行う
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.2)], p=0.7),
                # # ランダムに回転を行う
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=20)], p=0.7),

                # ランダムに画像を切り抜く
                transforms.RandomApply(
                    [transforms.RandomCrop(size=(435, 435), padding=(20, 20))], p=7.0),

                # 画像のサイズを変換する
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                # 指定された平均と標準偏差で画像を規格化する
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # ランダムに消去する
                transforms.RandomErasing(p=0.7, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0, inplace=False)
            ]),

            'test': transforms.Compose([
                # 画像のサイズを変換する
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)




class TripletDataset(Dataset):
    def __init__(self, datapath_dic, transform=None, phase='train'):
        self.datapath_dic = datapath_dic
        self.transform = transform
        self.phase = phase

        key_list = list(datapath_dic.keys())
        # print("key_list:", key_list)

        all_datapath = []
        all_datalabel = []
        for data_list in self.datapath_dic.values():
            all_datapath += data_list

        for i in key_list:
            for k in range(len(datapath_dic[i])):
                all_datalabel.append(i)

        self.all_datapath = all_datapath
        self.all_datalabel = all_datalabel

    def __len__(self):
        return len(self.all_datapath)

    def __getitem__(self, idx):
        anchor_label = self.all_datalabel[idx]

        # リストの要素のインデックス（何番目か）を取得
        idx_list = [i for i, x in enumerate(self.all_datalabel) if x == anchor_label]
        not_idx_list = [i for i, x in enumerate(self.all_datalabel) if not x == anchor_label]

        positive_image_list = []
        negative_image_list = []
        negative_image_label_list = []

        # print("idx_list:::::::", idx_list)

        for i in range(len(idx_list)):
            k = idx_list[i]
            positive_image_list.append(self.all_datapath[k])

        for i in range(len(not_idx_list)):
            k = not_idx_list[i]
            negative_image_list.append(self.all_datapath[k])
            negative_image_label_list.append(self.all_datalabel[k])

        # print("positive_image_list", positive_image_list)
        # print("negative_image_list", negative_image_list)
        # print("negative_image_label_list", negative_image_label_list)

        # negative用indexを作成
        random_idx = np.random.randint(0, len(not_idx_list))

        positive_image_path = random.choice(positive_image_list)
        negative_image_path = negative_image_list[random_idx]
        negative_label = negative_image_label_list[random_idx]

        anchor_np = np.array(Image.open(self.all_datapath[idx]))
        positive_np = np.array(Image.open(positive_image_path))
        negative_np = np.array(Image.open(negative_image_path))

        anchor_pil = Image.fromarray(np.uint8(anchor_np))
        positive_pil = Image.fromarray(np.uint8(positive_np))
        negative_pil = Image.fromarray(np.uint8(negative_np))

        anchor = self.transform(anchor_pil, self.phase)
        positive = self.transform(positive_pil, self.phase)
        negative = self.transform(negative_pil, self.phase)

        return anchor, positive, negative, anchor_label, negative_label


# 動作確認
if __name__ == '__main__':

    train_dic = Make_Datapath_Dic("train")
    test_dic = Make_Datapath_Dic("test")

    print("train_dic:", train_dic)
    print("train_dic.type:", type(train_dic))
    print("train_dic.len:", len(train_dic))
    print("test_dic:", test_dic)
    print("test_dic.type:", type(test_dic))
    print("test_dic.len:", len(test_dic))

    transform = ImageTransform()

    train_dataset = TripletDataset(train_dic, transform=transform, phase="train")
    test_dataset = TripletDataset(test_dic, transform=transform, phase="test")

    # print(len(train_dataset))
    # print(train_dataset[9])
    # print(type(train_dataset[9]))

    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (anchor, positive, negative, anchor_label, negative_label) in enumerate(train_dataloader):
        if batch_idx == 0:
            print("anchor_label.shape", anchor_label.shape)
            print("anchor_label", anchor_label)
            print("anchor_label[1]", anchor_label[0])
            print("negative_label.shape", negative_label.shape)
            print("negative_label", negative_label)
            print("negative_label[1]", negative_label[0])
            print("anchor.size:", anchor.size())
            print("positive.size:", positive.size())
            print("negative.size:", negative.size())
            # print("anchor[1]:", anchor[1])
            print("anchor[1]_type", type(anchor[1]))
            print("anchor[1]_shape:", anchor[1].shape)
            for i in range(batch_size):
                image = np.transpose(anchor[i], [1, 2, 0])
                image = image * 255
                image = np.array(image, dtype=np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                cv2.imshow("input_Image", image)
                cv2.waitKey(20)

    for batch_idx, (anchor, positive, negative, anchor_label, negative_label) in enumerate(test_dataloader):
        if batch_idx == 0:
            print("test_image_label.shape", anchor_label.shape)
            print("test_image_label", anchor_label)
            print("test_image_label[1]", anchor_label[0])
            print("test_image.size:", anchor.size())
            # print("test_image[1]:", anchor[1])
            print("test_image[1]_type", type(anchor[1]))
            print("test_image[1]_shape:", anchor[1].shape)
            # image = np.transpose(anchor[1], [1, 2, 0])
            # print("image_shape:", image.shape)
            # plt.imshow(image)
            # plt.show()

            print("Finish")

    print("Finish")




