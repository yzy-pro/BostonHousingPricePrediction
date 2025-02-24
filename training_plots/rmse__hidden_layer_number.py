import os
import pandas as pd  # 用于数据处理
import torch  # 用于深度学习
import matplotlib.pyplot as plt  # 用于绘制图形
from sklearn.model_selection import train_test_split  # 用于数据集拆分

# 设置环境变量，解决在使用 PyTorch 时，可能会遇到的 OpenMP 相关问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
train_data = pd.read_csv('../house-prices-advanced-regression-techniques/train.csv')  # 训练数据集
test_data = pd.read_csv('../house-prices-advanced-regression-techniques/test.csv')  # 测试数据集
processed_datas = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 合并训练集和测试集（去除 ID 列和目标标签列）

# 标准化数据
numeric_features = processed_datas.dtypes[processed_datas.dtypes != 'object'].index  # 获取所有数值型特征的列名
processed_datas[numeric_features] = processed_datas[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))  # 对数值特征进行标准化处理
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(0)  # 填充缺失值为 0

# 独热编码（One-Hot Encoding，OHE）处理类别特征
processed_datas = pd.get_dummies(processed_datas, dummy_na=True)  # 对所有类别变量进行 OHE，dummy_na=True 处理缺失值

# 将数据转换为 Tensor 类型，以便在 PyTorch 中使用
processed_datas = processed_datas.astype('float32')  # 转换数据类型为 float32
n_train = train_data.shape[0]  # 获取训练集的样本数量
processed_train_datas = torch.tensor(processed_datas[:n_train].values, dtype=torch.float32)  # 将训练集数据转换为 Tensor
processed_test_datas = torch.tensor(processed_datas[n_train:].values, dtype=torch.float32)  # 将测试集数据转换为 Tensor
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)  # 获取训练集标签并转换为 Tensor

# 定义一个多层感知机 (MLP) 模型
class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_units=256, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(in_features, hidden_units))  # 第一层输入到隐藏层

        # 添加剩余的隐藏层
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_units, hidden_units))  # 每个隐藏层都连接到前一个隐藏层

        self.output = torch.nn.Linear(hidden_units, 1)  # 最后一层输出预测值
        self.relu = torch.nn.ReLU()  # ReLU 激活函数

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))  # 对每一层的输出应用 ReLU 激活
        x = self.output(x)  # 输出层
        return x

# 损失函数：均方误差（MSE）
loss_fn = torch.nn.MSELoss()

# 计算训练日志 RMSE（均方根误差）
def log_rmse(net, features, labels, device):
    features = features.to(device)  # 将数据移到同一设备上
    labels = labels.to(device)  # 将标签移到同一设备上
    clipped_preds = torch.clamp(net(features), min=1.0)  # 对预测值做裁剪，确保预测值不小于 1
    rmse = torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels)))  # 对数 RMSE
    return rmse.item()

# 训练函数
def train(net, train_features, train_labels, valid_features, valid_labels, num_epochs, learning_rate, weight_decay, batch_size, device):
    train_loss, valid_loss = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)  # 数据加载器，按 batch_size 从训练数据中随机采样
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam 优化器

    for epoch in range(num_epochs):
        for X, y in train_iter:  # 对每个 batch 进行训练
            X, y = X.to(device), y.to(device)  # 将数据移到同一设备
            optimizer.zero_grad()  # 清除梯度
            loss = loss_fn(net(X), y)  # 计算当前 batch 的损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

        train_loss.append(log_rmse(net, train_features, train_labels, device))  # 计算并记录训练集的 RMSE
        valid_loss.append(log_rmse(net, valid_features, valid_labels, device))  # 计算并记录验证集的 RMSE

    return train_loss, valid_loss

# 主函数
if __name__ == "__main__":
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')  # 使用 XPU 或 CPU 设备
    print(f'Using device: {device}')

    # 设置要测试的隐藏层数值
    hidden_layer_values = [1, 2, 3, 4, 5, 6]  # 测试不同隐藏层的数值
    train_rmses = []
    valid_rmses = []

    # 拆分训练数据为训练集和验证集
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        processed_train_datas, train_labels, test_size=0.2, random_state=42)  # 80% 训练集，20% 验证集

    # 对不同的隐藏层数值执行训练并记录 RMSE
    for num_hidden_layers in hidden_layer_values:
        print(f"Running with num_hidden_layers={num_hidden_layers}")
        net = MLP(in_features=processed_train_datas.shape[1], num_hidden_layers=num_hidden_layers).to(device)
        train_loss, valid_loss = train(net, train_features, train_labels,
                                       valid_features, valid_labels,
                                       num_epochs=100,
                                       learning_rate=0.05, weight_decay=1e-4, batch_size=128, device=device)
        train_rmses.append(train_loss[-1])  # 只取最后一次训练 RMSE
        valid_rmses.append(valid_loss[-1])  # 只取最后一次验证 RMSE

        print(f"Final RMSE for num_hidden_layers={num_hidden_layers}: {train_loss[-1]:.6f}")

    # 创建文件夹用于保存图像
    os.makedirs("training_plots", exist_ok=True)

    # 绘制隐藏层数与最终 RMSE 的关系图
    plt.plot(hidden_layer_values, train_rmses, label='Training Loss')
    plt.plot(hidden_layer_values, valid_rmses, label='Validation Loss', linestyle='--')

    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Final RMSE')
    plt.title('RMSE vs Number of Hidden Layers')
    plt.legend()

    # 保存图像
    plt.savefig("training_plots/rmse_vs_hidden_layers (epoch=).png")
    plt.close()  # 关闭图像

    print("Plot saved as 'training_plots/rmse_vs_hidden_layers.png'.")
