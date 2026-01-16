import pandas as pd 
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
raw_data = load_diabetes()

# 转换成DataFrame
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df['Target'] = raw_data.target
# print(df.head())

x = df.iloc[:,:-1]
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

weights = pd.Series(model.coef_, index = x.columns)
print(weights.sort_values(ascending=False))