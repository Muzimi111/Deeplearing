from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import load_model

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
# print(df.head())
# 读取x,y
x = df.iloc[:,:-1]
y = df['target']
# 数据集划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# 将数据标签转换成独热编码
y_train = to_categorical(y_train, num_classes=2)
y_test_c = to_categorical(y_test, num_classes=2)
# 进行数据归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = load_model('breast_cancer_model.h5')

# 预测
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
# 打印详细分类报告
print("\n--- 详细分类报告 ---")
print(classification_report(y_test, y_pred, labels=[0,1], target_names=['恶性', '良性']))
