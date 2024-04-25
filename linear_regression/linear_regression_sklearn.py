import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 加载数据集
housing = fetch_california_housing()
x = housing.data
y = housing.target

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建线性回归对象
model = LinearRegression()

# 拟合模型
model.fit(X_train, Y_train)

# 使用测试集进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"R^2 分数: {r2:.2f}")

# 绘制预测结果与真实结果的对比图
plt.figure(figsize=(10, 6))  # 创建一个图形窗口，设置图形尺寸为宽度 10 英寸，高度 6 英寸

# 绘制散点图，x轴为真实值(Y_test)，y轴为预测值(y_pred)，alpha=0.5 表示散点的透明度为 0.5
plt.scatter(Y_test, y_pred, alpha=0.5)

# 绘制对角线参考线，参考线从原点(0, 0)到(x轴最大值，y轴最大值)，颜色为红色，线型为虚线
plt.plot([0, max(Y_test) + 1], [0, max(Y_test) + 1], color='red', linestyle='--')

# 设置坐标轴标签
plt.xlabel('真实值')  # x轴标签为“真实值”
plt.ylabel('预测值')  # y轴标签为“预测值”

# 设置图表标题
plt.title('线性回归模型预测结果')  # 图表标题为“线性回归模型预测结果”

# 显示图形
plt.show()

