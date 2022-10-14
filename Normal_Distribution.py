#numpy,scipy.statsからnorm,math,matplotlib.pyplotをインポート！
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

anchor_label = 5
negative_label = 8

#0から100まで、0.01間隔で入ったリストXを作る！
X = np.arange(0, 10, 1)
#確率密度関数にX,平均50、標準偏差20を代入
Y = norm.pdf(X, anchor_label, 3) * 5000

norm = norm.pdf(negative_label, anchor_label, 3) * 5000

print("norm", norm)


#x,yを引数にして、関数の色をr(red)に指定！カラーコードでも大丈夫です！
plt.plot(X, Y, color='r')
plt.xlabel("label")
plt.xlabel("margin")
plt.ylim(0, )
plt.savefig('figure.png')
plt.show()