import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# 定义一个自定义的训练集和测试集分割函数
def my_train_test_split(x, y, test_size=0.2, random_state=None):
    """
    实现数据集的分割功能，将给定的输入数据集划分为训练集和测试集两部分。

    此函数遵循常用的训练-测试划分策略，以实现模型训练过程中的验证和泛化性能评估。
    分割过程中，用户可以选择指定测试集所占数据集的比例以及随机种子，以确保分割结果的可重复性。

    参数:
    - x (np.ndarray): 特征数组，表示输入数据集中的特征向量集合。要求为二维数组，其中每一行代表一个样本，每一列代表一个特征。
    - y (np.ndarray): 目标数组，表示输入数据集中对应于每个样本的目标变量或标签。要求为一维数组，长度与特征数组的行数相同。
    - test_size (float, default=0.2): 测试集占总数据集的比例。取值范围为(0, 1)，默认值为0.2，即默认将数据集的20%作为测试集。
    - random_state (int, optional): 随机种子数。若指定此参数，将使用该种子初始化随机数生成器，确保在多次调用此函数时，对于相同的输入数据，分割结果保持一致。若未指定或设为None，将采用当前时间戳作为随机种子，可能导致每次分割结果的随机性。

    返回值:
    - x_train (np.ndarray): 训练集的特征数组。形状与输入的特征数组`x`相同，但仅包含被分配到训练集的样本特征。
    - x_test (np.ndarray): 测试集的特征数组。形状与输入的特征数组`x`相同，但仅包含被分配到测试集的样本特征。
    - y_train (np.ndarray): 训练集的目标数组。形状与输入的目标数组`y`相同，但仅包含被分配到训练集的样本目标变量。
    - y_test (np.ndarray): 测试集的目标数组。形状与输入的目标数组`y`相同，但仅包含被分配到测试集的样本目标变量。
    """
    # 如果设置了随机种子，就固定随机数生成器的种子
    if random_state is not None:
        np.random.seed(random_state)

    # 计算数据集中的样本总数
    n_samples = len(x)
    # 根据测试集比例计算测试集的样本数量
    test_samples = int(n_samples * test_size)

    # 对样本索引进行随机打乱，确保分割的随机性
    indices = np.random.permutation(n_samples)
    # 提取测试集的索引
    test_indices = indices[:test_samples]
    # 提取训练集的索引
    train_indices = indices[test_samples:]

    # 根据索引划分特征和目标数组
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test


# 定义一个简单的线性回归类
class my_linear_regression:
    """
    定义一个简单的多元线性回归模型类，用于拟合给定数据集并进行预测。

    该类实现了基于梯度下降法的训练过程，允许用户设置学习率和迭代次数以控制模型训练的收敛速度和精度。
    同时提供了属性来存储模型的权重和偏置，并提供了`fit`方法用于模型训练和`predict`方法用于新数据的预测。

    参数:
    learning_rate (float, default=0.01): 学习率，控制参数更新的速度。取值需大于0，默认值为0.01。
    n_iterations (int, default=1000): 迭代次数，模型训练的轮数。取值需为正整数，默认值为1000。

    属性:
    weights (np.ndarray): 模型的权重数组。存储了模型在训练过程中学习到的特征与目标变量之间的线性关系系数。
    bias (float): 模型的偏置项。反映了模型在所有特征值为0时的预测值。

    方法:
    fit(x, y): 训练线性回归模型。接受特征矩阵`x`和目标变量数组`y`作为输入，通过梯度下降法更新模型的权重和偏置。
    predict(x): 使用训练好的模型进行预测。接受特征矩阵`x`作为输入，返回对应的预测结果数组。
    """

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate  # 学习率
        self.n_iterations = n_iterations  # 迭代次数
        self.weights = None  # 权重初始化为None，将在训练前由fit方法初始化为0向量
        self.bias = None  # 偏置初始化为None，将在训练前由fit方法初始化为0
        # Adam优化器的参数
        self.m_w = None  # 初始化权重的一阶矩向量（动量）
        self.v_w = None  # 初始化权重的二阶矩向量（尺度）
        self.m_b = 0  # 初始化偏置的一阶矩向量（动量）
        self.v_b = 0  # 初始化偏置的二阶矩向量（尺度）
        self.beta1 = 0.9  # 动量衰减率，一般设为0.9
        self.beta2 = 0.999  # 尺度衰减率，一般设为0.999
        self.epsilon = 1e-8  # 数值稳定性常数，防止除以零

    def fit(self, x, y):
        """
        训练线性回归模型。

        此方法接收特征矩阵`x`和目标变量数组`y`作为输入，根据给定的学习率和迭代次数，通过梯度下降法更新模型的权重和偏置。

        参数:
        x (np.ndarray): 特征矩阵，形状为(n_samples, n_features)，其中n_samples为样本数量，n_features为特征数量。
        y (np.ndarray): 目标变量数组，形状为(n_samples,),与特征矩阵`x`的行数相同，表示每个样本的目标值。

        注意：此方法会覆盖之前训练得到的模型权重和偏置。
        """
        n_samples, n_features = x.shape  # 样本数和特征数
        self.weights = np.zeros(n_features)  # 权重初始化为0
        self.bias = 0  # 偏置初始化为0
        self.m_w = np.zeros(n_features)  # 初始化权重的一阶矩为0
        self.v_w = np.zeros(n_features)  # 初始化权重的二阶矩为0
        for i in range(self.n_iterations):
            y_pred = self.predict(x)
            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))  # 计算权重的梯度
            db = (1 / n_samples) * np.sum(y_pred - y)  # 计算偏置的梯度

            # 使用Adam算法更新权重
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw  # 更新一阶矩估计
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)  # 更新二阶矩估计
            m_hat_w = self.m_w / (1 - self.beta1 ** (i + 1))  # 修正一阶矩估计的偏差
            v_hat_w = self.v_w / (1 - self.beta2 ** (i + 1))  # 修正二阶矩估计的偏差
            self.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)  # 更新权重

            # 使用Adam算法更新偏置
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db  # 更新一阶矩估计
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)  # 更新二阶矩估计
            m_hat_b = self.m_b / (1 - self.beta1 ** (i + 1))  # 修正一阶矩估计的偏差
            v_hat_b = self.v_b / (1 - self.beta2 ** (i + 1))  # 修正二阶矩估计的偏差
            self.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)  # 更新偏置

            if i % 100 == 0:  # 每100次迭代打印一次当前权重和偏置
                print(f"Iteration {i}: Weights={self.weights}, Bias={self.bias}")

        return self

    def predict(self, x):
        """
        使用训练好的模型进行预测。

        此方法接收特征矩阵`x`作为输入，利用已训练好的模型权重和偏置计算出对应的预测值数组。

        参数:
        x (np.ndarray): 输入数据，形状为(n_samples, n_features)，与训练时使用的特征矩阵形状相同。

        返回:
        np.ndarray: 预测结果数组，形状为(n_samples,), 表示对输入数据中每个样本的预测值。
        """
        return np.dot(x, self.weights) + self.bias


# 定义均方误差函数
def my_mean_squared_error(y_true, y_pred):
    """
    计算均方误差（MSE），作为评估模型预测性能的一种指标。

    均方误差是预测值与真实值之差的平方的平均值，越小表示模型预测效果越好。

    参数:
    y_true (np.ndarray): 真实值数组，形状为(n_samples,), 表示实际观测到的目标变量值。
    y_pred (np.ndarray): 预测值数组，形状为(n_samples,), 表示模型对每个样本的预测值。

    返回:
    float: 均方误差（MSE）值。
    """
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse


# 定义R²得分函数
def my_r2_score(y_true, y_pred):
    """
    计算R²得分（决定系数），作为评估模型预测性能的另一种指标。

    R²得分表示模型预测值与真实值之间关系的强度，取值范围为[0, 1]，越接近1表示模型拟合程度越高。

    参数:
    y_true (np.ndarray): 真实值数组，形状为(n_samples,), 表示实际观测到的目标变量值。
    y_pred (np.ndarray): 预测值数组，形状为(n_samples,), 表示模型对每个样本的预测值。

    返回:
    float: R²得分值。
    """
    # 计算真实值的平均值
    mean_y_true = np.mean(y_true)
    # 计算总平方和，即真实值与真实值平均值之差的平方和
    total_sum_of_squares = np.sum((y_true - mean_y_true) ** 2)
    # 计算残差平方和，即预测值与真实值之差的平方和
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    # 计算R²得分
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2

# 加载加利福尼亚州房屋数据集
housing = fetch_california_housing()

# 提取数据集中的特征数组 x 和目标数组 y
x = housing.data
y = housing.target

# 打印特征数组 x 和目标数组 y 的形状，以便了解数据集的基本结构
print("特征数组 x 的形状:", x.shape)
print("目标数组 y 的形状:", y.shape)

# 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 划分数据集为训练集和测试集
# 使用自定义的 my_train_test_split 函数，按照 80% 训练集和 20% 测试集的比例进行分割
# 并指定随机种子为 42，以确保结果的可复现性
x_train, x_test, y_train, y_test = my_train_test_split(x, y, test_size=0.2, random_state=42)

# 创建线性回归模型实例
# 使用自定义的 my_linear_regression 类创建线性回归模型对象，采用默认的学习率和迭代次数
model = my_linear_regression(learning_rate=0.01, n_iterations=500)

# 训练线性回归模型
model.fit(x_train, y_train)

# 使用训练好的模型对测试集进行预测
y_pred = model.predict(x_test)

# 计算模型在测试集上的均方误差（MSE）和 R² 得分
# 分别使用自定义的 my_mean_squared_error 和 my_r2_score 函数进行计算
mse = my_mean_squared_error(y_test, y_pred)
r2 = my_r2_score(y_test, y_pred)

# 打印模型的均方误差和 R² 得分，以评估模型性能
print(f"模型均方误差 (MSE): {mse:.2f}")
print(f"模型 R² 得分: {r2:.2f}")

# 绘制真实值与预测值的散点图以及拟合线
# 使用 matplotlib 库绘制散点图，展示测试集上真实值 y_test 与预测值 y_pred 之间的关系
plt.figure(figsize=(10, 6))

# 绘制散点图，蓝色散点表示真实值与预测值的对应关系，设置透明度为 0.5
plt.scatter(y_test, y_pred, alpha=0.5)

# 绘制红色虚线作为理想拟合线，即 y = x，表示完美预测的情况
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')

# 设置图表的标题、x 轴和 y 轴的标签
plt.title('my linear regression')
plt.xlabel('True Values')
plt.ylabel('Predictions')

# 显示绘制完成的图表
plt.show()
