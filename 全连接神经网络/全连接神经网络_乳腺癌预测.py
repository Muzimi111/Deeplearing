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
# print(x_test)

# 构建神经网络模型
model = keras.Sequential()
model.add(Dense(units=16, input_dim=30, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)
model.save('breast_cancer_model.h5')

# 绘制训练过程中的损失和准确率曲线
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型损失曲线')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='训练集准确率')
plt.plot(history.history['val_accuracy'], label='验证集准确率')
plt.title('模型准确率曲线')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

