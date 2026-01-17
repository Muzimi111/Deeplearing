import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载手写数字数据集
digits = load_digits()
x = digits.data
y = digits.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 训练多层感知机模型
hiden_layer_sizes = (100, 50)  # 两个隐藏层，分别有100和50个神经元
mlp = MLPClassifier(hidden_layer_sizes=hiden_layer_sizes, activation='relu',max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)
# 预测
y_pred = mlp.predict(x_test)
# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率为: {accuracy:.4f}")
# 6. 看看训练过程 (Loss曲线)
# 神经网络不仅看结果，还要看它是怎么“学”的
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve (Learning Process)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# 7. 打印详细报告
print("\n--- 详细分类报告 ---")
print(classification_report(y_test, y_pred))