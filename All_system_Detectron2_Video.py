from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import sys
import time

import requests
from requests.auth import HTTPDigestAuth
import io
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

# Setting a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TripletNet().to(device)
# 210920_adaptive_Tripletloss/adaptive_3_10000epc
model_dir = "output/210920_adaptive_Tripletloss/adaptive_3_10000epc/"
model_weights = model_dir + "10000_epoch.pth"
model.load_state_dict(torch.load(model_weights))

# エラー用のしきい値の設定
threshold = 3000000

# Datasetの準備
batch_size = 4
train_dic = Make_Datapath_Dic("train")
transform = ImageTransform()
train_kNN_dataset = TripletDataset(train_dic, transform=transform, phase="test")
train_kNN_dataloader = DataLoader(train_kNN_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Basis Data Embedding for 128 dimension space
kNN_basis_np, kNN_basis_label_np = Basis_Embedding(model, train_kNN_dataloader)

print("kNN_basis_np:", kNN_basis_np)
print("kNN_basis_np.shape:", kNN_basis_np.shape)
print("kNN_basis_label_np:", kNN_basis_label_np)
print("kNN_basis_label_np.shape:", kNN_basis_label_np.shape)

# 動画の読み込み
# video_dir = "/mnt/sdb1/Kitsukawa Windows HDD/workspace/programs/pycharm/Sample_py36/sample_py36/QWATCH/desktopPC_Dataset/"
# cap = cv2.VideoCapture(video_dir + "sample3_1.mp4")
cap = cv2.VideoCapture("input/demo1.mp4")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

# True or False
print(cap.isOpened())
print("Frame_width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame_height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame_rate:", cap.get(cv2.CAP_PROP_FPS))
print("Frame_count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video_time:", cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))


# Video save settings
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter('output/demo3.mp4', fourcc, 5.5, (width, height)) #18.1

def Transform_Qwatch(input):
    img_bin = io.BytesIO(input)
    img_pil = Image.open(img_bin)
    img_np = np.asarray(img_pil)
    image = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    return image

def Transform_for_Model(image):
    transform1 = transforms.Compose([
        # 画像のサイズを変換する
        transforms.Resize((448, 448)),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        # 指定された平均と標準偏差で画像を規格化する
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform1(image)

    # img1 = np.asarray(image)
    # plt.imshow(img1)
    # plt.show()

    image = transform2(image)

    # 次元の追加，[チャンネル数，高さ，幅]から[バッチ数（１），チャンネル数，高さ，幅]
    image = image[np.newaxis, :, :, :]
    # print(image.shape)
    return image

# モデルの設定
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95 # set threshold for this model
cfg.MODEL.WEIGHTS = "/mnt/sdb1/Kitsukawa Windows HDD/workspace/programs/detectron2_tutorial/desktopPC1/model_final.pth"  #hitachi_output_1103, desktopPC1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


# 推論(predictor)+
predictor = DefaultPredictor(cfg)
delay = 1


color_list = [(50, 199, 90), (50, 177, 199), (209, 166, 56), (56, 79, 209), (209, 56, 184), (209, 67, 56), (198, 86, 227), (247, 235, 99), (255, 110, 0), (80, 110, 250), (180, 25, 70), (60, 270, 150)]
color_list2 = [(50, 177, 199)]
y_add = 5

# OpenCVアルコマーカーの設定
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

sum_frame = 0
step_list = list(range(20))
step_fix = 0
step_all = []


if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    if ret:
        start = time.time()


        start = time.time()
        MetadataCatalog.get("my_train_data").set(thing_classes=["cart", "human"])
        my_metadata = MetadataCatalog.get("my_train_data")

        outputs = predictor(frame)

        # v = Visualizer(frame[:, :, ::-1], metadata=my_metadata, scale=1.0)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('Results', v.get_image()[:, :, ::-1])

        instances = outputs["instances"].to("cpu")
        # print(type(instances))
        # print("Number of objects:", len(instances))

        frame_raw = frame

        for i in range(len(instances)):

            # print("number=", i)

            # ボックス四点の座標を取得
            box = instances.get("pred_boxes")
            # print("box_type:", type(box))
            # print("box:", box)
            box = box.tensor.to("cpu").numpy()[i]  # 何故か次元が一つ多い
            # print("box_type:", type(box))
            # print("box:", box)
            x1, y1 = box[0], box[1]
            x2, y2 = box[2], box[3]
            # print("x1, y1 :", x1,  y1)
            # print("x2, y2 :", x2,  y2)

            # 画像の切り出し
            # frame[top : bottom, left : right]
            a = 30
            frame_cutout = frame[int(y1):int(y2), int(x1):int(x2)]
            height = frame_cutout.shape[0]
            width = frame_cutout.shape[1]
            frame_ratio = height / width
            # frame_cutout = cv2.resize(frame_cutout, (int(width * 2), int(height * 2)))

            # transform for model
            frame_cutout_model = Transform_for_Model(frame_cutout)
            # Progress Estimation
            step_number, distance, img_embedding_np = Estimate_Image(model, frame_cutout_model, kNN_basis_np, kNN_basis_label_np)
            print("Step:", step_number)


            pix_show = 400

            frame_cutout = cv2.resize(frame_cutout, (int(pix_show), int(pix_show * frame_ratio)))

            cv2.imshow("frame_cutout", frame_cutout)

            cv2.putText(frame, "desktopPC", (int(x1), int(y1)-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

            sum_frame += 1

            # バウンディングボックスを表示
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list2[0], 5)

            # 確率を表示
            class_prob = instances.get("scores").numpy()[i]
            str_class_prob = "{:.2f}".format(class_prob)
            cv2.putText(frame, str_class_prob, (int(x1), int(y1 - y_add)), cv2.FONT_HERSHEY_PLAIN, 2, color_list2[0], 3)

        elapsed_time = time.time() - start
        # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        fps = "{:.1f}".format(1 / elapsed_time)
        frame = cv2.resize(frame, (1920, 1080))
        frame = cv2.putText(
            frame, "Frame per Second: %s" % fps, (1400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 5)

        if distance < threshold:
            frame = cv2.putText(
                frame, "Estimated Step: %s" % step_number, (1450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 5)
            step_list.append(step_number)
            step_list.pop(0)
            step_all.append(step_number)

            # 10フレーム連続で同じステップと判定したら更新
            if step_list == [step_list[0]] * len(step_list):
                if step_number > step_fix:
                    step_fix = step_number

        else:
            frame = cv2.putText(
                frame, "ERROR: (%s)" % step_number, (1450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 5)


        frame = cv2.putText(
            frame, str(step_fix), (1600, 500), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 10)


        cv2.imshow("frame", frame)

        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


writer.release()
cv2.destroyAllWindows()

x_data = list(range(len(step_all)))
y_data = step_all

# グラフの表示
plt.figure()
plt.plot(x_data, y_data, color='black', markersize=3, markeredgewidth=3, marker="x", linestyle='None')
# plt.plot(x_epoch_data, y_test_loss_data, color='orange', label='Test')
plt.xlabel('Frame')
plt.ylabel('Step')
plt.grid()
plt.title('Estimated Results')
plt.savefig("output//result2.png")

plt.show()


print("--------FINISH--------")
