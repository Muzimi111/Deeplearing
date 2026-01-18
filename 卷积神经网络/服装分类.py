import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']    
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']
'''
print(f"训练集形状: {train_images.shape}, 测试集形状: {test_images.shape}")
plt.figure(figsize = (10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
# --------数据预处理--------
# 1. 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 2. 调整形状以适应卷积神经网络输入要求
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
print(f"调整后训练集形状: {train_images.shape}, 测试集形状: {test_images.shape}")

# --------构建卷积神经网络模型--------
# 一个经典的 CNN 结构通常是：“卷积 -> 池化 -> 卷积 -> 池化 -> ... -> 拉平 -> 全连接(MLP)”
model = models.Sequential()
# 添加卷积层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加池化层
model.add(layers.MaxPooling2D((2, 2)))
# 添加第二个卷积层和池化层,用更多过滤器提取更复杂的特征
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 分类部分
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 10类输出,对应10个类别
# 查看模型结构摘要
model.summary()

# --------编译和训练模型--------
'''
Optimizer (优化器): adam 是目前最常用的，它能自动调整学习率。
Loss (损失函数)：因为标签是整数 (0-9)，用 sparse_categorical_crossentropy。如果标签是 One-hot 编码的，就用 categorical_crossentropy。
Metrics (指标)：关心准确率 accuracy。
'''
# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
print("模型训练完成。")

# --------评估模型与可视化--------
# 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n测试集准确率: {test_acc:.2%}")
# 绘制训练过程中的损失和准确率曲线
plt.figure(figsize=(12, 5))
# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('模型损失曲线')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练集准确率')
plt.plot(history.history['val_accuracy'], label='验证集准确率')
plt.title('模型准确率曲线')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()