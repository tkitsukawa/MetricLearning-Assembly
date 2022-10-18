# MetricLearning-Assembly

人手による製品の組み立て作業において、カメラ情報から作業進捗を推定するシステム。

組立作業者を見て進捗を推定するのではなく、製品状態を基に進捗ステップを推定する。
進捗推定は少ない学習データでも特徴を抽出できるようにするためDeep Metric Learningを利用する。

ネットワークのモデル構造は以下である。

![image](https://user-images.githubusercontent.com/64144764/196383560-72e829a7-45b8-48c4-95e0-85b1e3cdc6fc.png)
