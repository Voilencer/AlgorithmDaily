import torch
import numpy as np
import matplotlib.pyplot as plt



"""如何用pytorch拟合线性模型
model:  y = 5 * x + 7


"""

# 训练数据
x = np.random.rand(256)
y = x * 5 + 7 + np.random.randn(256) / 4

# Linear: y = w * x + b
# 输入1，输出1
model = torch.nn.Linear(1, 1)

# 损失函数
criterion = torch.nn.MSELoss()

# 优化器 SGD
# SGD ,每一次迭代计算mini-batch的梯度
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 3000
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

for i in range(epochs):
    inputs = torch.from_numpy(x_train)  # 转化为tensor
    labels = torch.from_numpy(y_train)
    outputs = model(inputs) # 模型预测
    optim.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()    # 更新参数
    if i % 99 == 0:
        print("epoch {}, loss:{:.4f}".format(i, loss.data.item()))

[w, b] = model.parameters()
print(w.item(), b.item())


predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label = 'data', alpha = 0.3)
plt.plot(x_train, predicted, label = 'predicted', alpha = 1)
plt.legend()
plt.show()






