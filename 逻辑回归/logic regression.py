from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

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

# 进行数据归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# print(x_test)

# 逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train, y_train)

# print('w:',lr.coef_)
# print('b:',lr.intercept_)

# 模型测试
pre_result = lr.predict(x_test)
pre_result_p = lr.predict_proba(x_test)

# 获取恶性肿瘤的概率
pre_list = pre_result_p[:,1] 
threshold = 0.5
result = []
for p in pre_list:
    if p > threshold:
        result.append(1)
    else:
        result.append(0)
# print(result)

# 输出指标
report = classification_report(y_test,result,labels=[0,1],target_names=['良性肿瘤','恶性肿瘤'])
print(report)