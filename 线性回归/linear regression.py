# 定义数据集
# 定义数据特征
x_data = [1, 2, 3]
# 定义数据标签
y_data = [2, 4, 6]
# 初始化参数w
w = 4
# 定义线性回归模型
def forward(x):
    return x*w

# 定义损失函数
def cost(xs, ys):
    cost_value = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost_value += (y-y_pred)**2
    return cost_value/len(xs)

# 定义梯度计算函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2*x*(x*w-y)
    return grad / len(xs)

# 开始训练
for epoch in range (100):

    w = w - 0.01 * gradient(x_data, y_data)
    cost_val = cost(x_data, y_data)
    print("const:",epoch,"loss:",cost_val,"w:",w)