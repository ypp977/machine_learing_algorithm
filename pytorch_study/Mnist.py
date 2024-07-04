import torch
import pickle
import gzip
import requests
import numpy as np

from pathlib import Path
from matplotlib import pyplot


print(torch.__version__)
# 读取mnist数据集
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL="http://deeplearning.net/data/mnist"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(f"{URL}/{FILENAME}").content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

pyplot.show()
print(f"x_train shape:{x_train.shape}")
print(f"y_train[:10]:{y_train[:10]}, y_train数据类型:{type(y_train)}")

# 将数据转为tensor格式，才能参与后续建模训练
x_train, y_train, x_valid, y_valid = map(
    lambda x: torch.tensor(x).float(), (x_train, y_train, x_valid, y_valid)
)
y_train, y_valid = y_train.long(), y_valid.long()  # 转换y_train和y_valid为long类型

# n样本个数，c特征个数
n, c = x_train.shape
print(f"n:{n}, c:{c}")
print(f"x_train:{x_train}\ny_train:{y_train}")
print(f"x_train.shape:{x_train.shape}")
print(f"y_train.min():{y_train.min()}\ny_train.max():{y_train.max()}")

#常用函数
import torch.nn.functional as F

loss_func = F.cross_entropy # 交叉熵损失函数

def model(xb):
    # 使用矩阵乘法 (@ 或 .mm()) 将输入数据xb与模型权重weights相乘，
    # 然后加上偏置bias。这是典型的线性模型计算过程。
    return xb @ weights + bias

bs = 64
xb = x_train[0:bs] # 从x_train中取64个样本
yb = y_train[0:bs] # 从y_train中取64个样本

weights = torch.randn(784, 10, requires_grad=True) # 随机初始化权重
bias = torch.zeros(10, requires_grad=True)

print(f"loss_func(model(xb),yb):{loss_func(model(xb), yb)}")


from torch import nn  # 导入torch的神经网络模块

class Mnist_nn(nn.Module):  # 定义一个继承自nn.Module的类
    def __init__(self):
        super().__init__()  # 显式调用父类初始化
        # 定义网络层
        self.hidden1 = nn.Linear(784, 128)  # 输入层到第一个隐藏层
        self.hidden2 = nn.Linear(128, 256)  # 第一个隐藏层到第二个隐藏层
        self.out = nn.Linear(256, 10)  # 第二个隐藏层到输出层
        self.dropout = nn.Dropout(0.5)  # dropout层，丢弃50%的神经元

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # 对第一层隐藏层应用ReLU激活函数
        x = self.dropout(x)  # 对第一层输出应用dropout
        x = F.relu(self.hidden2(x))  # 对第二层隐藏层应用ReLU激活函数
        x = self.dropout(x)  # 对第二层输出应用dropout
        x = self.out(x)  # 最后应用输出层
        return x  # 返回网络的最终输出

net = Mnist_nn()  # 实例化神经网络
print(net)  # 打印网络结构

for name, param in net.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
    
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

def getdata(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)
    )
    
import numpy as np
def fit(steps,model,loss_func,opt,train_dl,valid_dl):
    for step in range(steps):
        model.train()
        for xb,yb in train_dl:
            loss_batch(model,loss_func,xb,yb,opt)
        
        model.eval()
        with torch.no_grad():
            losses,nums = zip(
                *[loss_batch(model,loss_func,xb,yb) for xb,yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)
        print(f"Step: {step}, Validation loss: {val_loss}")
        
from torch import optim
def get_model():
     model = Mnist_nn()
     return model,optim.SGD(model.parameters(), lr=0.001)

def loss_batch(model,loss_func,xb,yb,opt=None):
    loss = loss_func(model(xb),yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    return loss.item(),len(xb)