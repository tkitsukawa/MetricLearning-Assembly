import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG16Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Block1 畳み込み層×2,BatchNormalization,Maxプーリング層
        # nn.Conv2d(入力チャンネル数,出力チャンネル数,カーネルサイズ,パディングの量)
        self.block1_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.block1_batch = nn.BatchNorm2d(64)
        self.block1_pool = nn.MaxPool2d(2, stride=2)

        # Block2 畳み込み層×2,BatchNormalization,Maxプーリング層
        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.block2_batch = nn.BatchNorm2d(128)
        self.block2_pool = nn.MaxPool2d(2, stride=2)

        # Block3 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.block3_batch = nn.BatchNorm2d(256)
        self.block3_pool = nn.MaxPool2d(2, stride=2)

        # Block4 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.block4_batch = nn.BatchNorm2d(512)
        self.block4_pool = nn.MaxPool2d(2, stride=2)

        # Block5 畳み込み層×4,BatchNormalization,Maxプーリング層
        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_batch = nn.BatchNorm2d(512)
        self.block5_pool = nn.MaxPool2d(2, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)

        # # 全結合層
        # self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        # self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        # self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, x):
        # Block1
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        x = F.relu(self.block1_batch(x))
        x = self.block1_pool(x)

        # Block2
        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = F.relu(self.block2_batch(x))
        x = self.block2_pool(x)

        # Block3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        x = F.relu(self.block3_batch(x))
        x = self.block3_pool(x)

        # Block4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_conv3(x))
        x = F.relu(self.block4_batch(x))
        x = self.block4_pool(x)

        # Block5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))
        x = F.relu(self.block5_batch(x))
        x = self.block5_pool(x)

        x = self.avgpool(x)

        # # 全結合層
        # x = x.view(-1, 512*7*7)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.dropout(x)
        # x = self.fc3(x)

        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_dim=128, num_class=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(in_features=128*28*28, out_features=128)

        # self.avgpool = nn.AvgPool2d(2)
        # self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.maxpool(x)
        x = x.view(-1, 128*28*28)
        x = F.relu(self.fc(x))
        # x = self.dropout(x)
        # x = self.classifier(x)

        return x



class TripletNet_v2(nn.Module):
    def __init__(self, embedding_dim=128, num_class=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(in_features=128*28*28, out_features=16)

        # self.avgpool = nn.AvgPool2d(2)
        # self.classifier = nn.Linear(embedding_dim, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.maxpool(x)
        x = x.view(-1, 128*28*28)
        x = F.relu(self.fc(x))
        # x = self.dropout(x)
        # x = self.classifier(x)

        return x


class TripletResNet(nn.Module):
    def __init__(self, metric_dim=128):
        super(TripletResNet, self).__init__()
        resnet = torchvision.models.__dict__['resnet18'](pretrained=True)
        for params in resnet.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, out_features=128)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    # def forward(self, x):
    #     x = self.model(x)
    #     x = x.view(x.size(0), -1)
    #     # metric = self.fc(x)
    #     metric = F.normalize(self.fc(x))
    #     return metric
