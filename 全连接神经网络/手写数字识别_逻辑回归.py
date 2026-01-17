from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 加载手写数字数据集
digits = load_digits()
'''
# 1. 看看它的真实身份
print(f"它的类型是: {type(digits)}") 
# 输出: <class 'sklearn.utils._bunch.Bunch'>

# 2. 看看它里面装了哪些“抽屉”（Keys）
print(f"它包含的属性: {digits.keys()}")
# 输出类似: dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
'''
# 准备数据
x = digits.data
y = digits.target
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 归一化,这里用MinmaxScaler是因为像素值是固定范围的（0-16）
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 训练模型
model = LogisticRegression(max_iter=3000)
model.fit(x_train, y_train)

# 预测
y_pred = model.predict(x_test)
# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率为: {accuracy:.4f}")
print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))  
# 我们从测试集中随机挑 6 张图
# 注意：我们要去 digits.images 里找原始图像用来画图
# 但预测时要用 X_test_scaled
num_samples = 6
indices = np.random.choice(len(x_test), num_samples, replace=False)

plt.figure(figsize=(10, 5))

for i, index in enumerate(indices):
    plt.subplot(2, 3, i + 1)
    
    # 1. 画出原始图片 (需要把拉直的 64个特征 变回 8x8 才能画)
    # 这里的 reshape(8, 8) 就是把条状变回方块
    image_data = x_test[index].reshape(8, 8)
    plt.imshow(image_data, cmap='gray', interpolation='nearest')
    
    # 2. 获取预测结果和真实结果
    pred_label = y_pred[index]
    true_label = y_test[index]
    
    # 3. 标题：如果预测对了用绿色，错了用红色
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"Pred: {pred_label} | True: {true_label}", color=color)
    plt.axis('off') # 关掉坐标轴刻度，好看一点

plt.tight_layout()
plt.show()