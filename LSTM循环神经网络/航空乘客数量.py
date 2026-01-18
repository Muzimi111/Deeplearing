import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 读取数据
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1]) # 只读取乘客数量列

# 数据预处理,转换成浮点数
data = df.values.astype('float32')
# print(f"原始数据形状: {data.shape}")
# print(f"前5行数据:\n{data[:5]}")
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 切分数据集
'''
为什么不用sklearn的train_test_split呢？
因为时间序列数据有时间顺序，直接打乱会破坏时间依赖性
'''
train_size = int(len(data)* 0.8)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
# print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")

# create dataset是用来做什么的？
# 用于将一维时间序列数据转换为适合LSTM训练的二维数据格式
# look_back表示用多少个时间步的数据来预测下一个时间步,为什么是1？
# 因为假设当前时间点的乘客数量只依赖于前一个时间点的数量
def create_dataset(dataset, look_back=1):
    X , y = [], []
    for i in range(len(dataset)- look_back - 1):
        # 提取从i开始的look_back个时间步的数据作为输入特征
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10
# 生成训练集和测试集
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
# 调整输入数据形状为 [样本数, 时间步长, 特征数]
# LSTM需要3D输入,这里特征数是1因为每个时间步只有一个特征:乘客数量
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# print(f"X_train形状: {X_train.shape}")

# 构建LSTM模型
model = Sequential()
# 添加一个LSTM层，50个神经元，输入形状为(look_back, 1)
# return_sequences=True: 意思是这层 LSTM 会为每个时间步都输出一个结果（逐步发言）。
# 只要下一层还是LSTM层，就需要设置为True
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
# return_sequences=False: 意思是这层 LSTM 只吐出最后一个时间步的结果（总结性发言）。
model.add(LSTM(50, return_sequences=False))
# 添加输出层，1个神经元用于预测下一个时间步的乘客数量
# 为什么全连接层是25个神经元？
# 为了增加模型的表达能力，使模型能够学习更复杂的非线性关系
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

# 编译模型，使用均方误差损失函数和adam优化器
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
# verbose=2表示每个epoch输出一行日志
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
# 反归一化预测结果
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])
 
# 准备训练集预测结果的绘图数据
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
# 从 look_back 开始填入预测数据
train_plot[look_back:len(train_predict)+look_back, :] = train_predict

# 准备测试集预测结果的绘图数据
test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
# 也就是在训练集之后接着画
test_plot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict

# 画图
plt.figure(figsize=(10, 6))
# 1. 原始数据 (蓝色)
plt.plot(scaler.inverse_transform(data), label='True Data')
# 2. 训练集预测 (橙色)
plt.plot(train_plot, label='Train Prediction')
# 3. 测试集预测 (绿色) 
plt.plot(test_plot, label='Test Prediction')

plt.title("LSTM Time Series Prediction")
plt.legend()
plt.show()