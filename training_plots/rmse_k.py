import os
import pandas  # 用于数据处理
import torch  # 用于深度学习
import matplotlib.pyplot as plt  # 用于绘制图形

# 设置环境变量，解决在使用 PyTorch 时，可能会遇到的 OpenMP 相关问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
train_data = pandas.read_csv('../house-prices-advanced-regression-techniques'
                             '/train.csv')  # 训练数据集
test_data = pandas.read_csv('../house-prices-advanced-regression-techniques'
                            '/test.csv')  # 测试数据集
processed_datas = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:,
                                                           1:]))  # 合并训练集和测试集（去除 ID 列和目标标签列）

# 标准化数据
numeric_features = processed_datas.dtypes[
    processed_datas.dtypes != 'object'].index  # 获取所有数值型特征的列名
processed_datas[numeric_features] = processed_datas[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))  # 对数值特征进行标准化处理
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(
    0)  # 填充缺失值为 0

# 独热编码（One-Hot Encoding，OHE）处理类别特征
processed_datas = pandas.get_dummies(processed_datas,
                                     dummy_na=True)  # 对所有类别变量进行 OHE，dummy_na=True 处理缺失值

# 将数据转换为 Tensor 类型，以便在 PyTorch 中使用
processed_datas = processed_datas.astype('float32')  # 转换数据类型为 float32
n_train = train_data.shape[0]  # 获取训练集的样本数量
processed_train_datas = torch.tensor(processed_datas[:n_train].values,
                                     dtype=torch.float32)  # 将训练集数据转换为 Tensor
processed_test_datas = torch.tensor(processed_datas[n_train:].values,
                                    dtype=torch.float32)  # 将测试集数据转换为 Tensor
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                            dtype=torch.float32)  # 获取训练集标签并转换为 Tensor


# 定义一个多层感知机 (MLP) 模型
class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_units=256, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(
            torch.nn.Linear(in_features, hidden_units))  # 第一层输入到隐藏层

        # 添加剩余的隐藏层
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_units, hidden_units))  # 每个隐藏层都连接到前一个隐藏层

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
    rmse = torch.sqrt(
        loss_fn(torch.log(clipped_preds), torch.log(labels)))  # 对数 RMSE
    return rmse.item()


# 训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, device):
    train_loss, test_loss = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)  # 数据加载器，按 batch_size 从训练数据中随机采样
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # Adam 优化器

    for epoch in range(num_epochs):
        for X, y in train_iter:  # 对每个 batch 进行训练
            X, y = X.to(device), y.to(device)  # 将数据移到同一设备
            optimizer.zero_grad()  # 清除梯度
            loss = loss_fn(net(X), y)  # 计算当前 batch 的损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

        train_loss.append(log_rmse(net, train_features, train_labels,
                                   device))  # 计算并记录训练集的 RMSE
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels,
                                      device))  # 计算并记录验证集的 RMSE

    return train_loss, test_loss


# 获取 K 折交叉验证的数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # 每一折的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 分割数据集
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part  # 验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part  # 训练集
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 执行 K 折交叉验证，返回每个 k 对应的训练和验证 RMSE
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, num_hidden_layers, device):
    valid_rmses = []
    train_rmses = []  # 记录每个 k 折训练集的 RMSE
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = MLP(in_features=X_train.shape[1],
                  num_hidden_layers=num_hidden_layers).to(device)
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate,
                                       weight_decay, batch_size,
                                       device)  # 训练模型并获取训练和验证损失

        # 只保存最后一次折的 RMSE
        train_rmses.append(train_loss[-1])  # 记录训练集最后一次 RMSE
        valid_rmses.append(valid_loss[-1])  # 记录验证集最后一次 RMSE

    return train_rmses, valid_rmses  # 返回训练和验证的 RMSE


# 主函数
if __name__ == "__main__":
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')  # 使用
    # CUDA 设备
    print(f'Using device: {device}')

    # 设置要测试的 k 值
    k_values = [2, 3, 4, 5, 6, 7, 8]  # 确保 k 值是整数
    train_rmses = []
    valid_rmses = []

    # 对不同的 k 值执行 K 折交叉验证
    for k in k_values:
        print(f"Running K-fold with k={k}")
        tr_rmse, val_rmse = k_fold(k, processed_train_datas, train_labels,
                                   num_epochs=50,
                                   learning_rate=0.05, weight_decay=1e-4,
                                   batch_size=128,
                                   num_hidden_layers=3, device=device)
        train_rmses.append(tr_rmse[-1])  # 只取每个 k 折的最后一次训练 RMSE
        valid_rmses.append(val_rmse[-1])  # 只取每个 k 折的最后一次验证 RMSE
        print(f"Final Validation RMSE for k={k}: {val_rmse[-1]:.6f}")

    # 创建文件夹用于保存图像
    os.makedirs("training_plots", exist_ok=True)

    # 绘制 K 值与最终验证 RMSE 的关系图
    plt.plot(k_values, valid_rmses, label='Validation Loss')
    plt.plot(k_values, train_rmses, label='Training Loss')

    # 设置 x 轴的刻度为整数
    plt.xticks(k_values)  # 强制设置 x 轴为整数

    plt.xlabel('K (Number of folds)')
    plt.ylabel('Final RMSE')
    plt.title('Final Validation RMSE vs K-folds')
    plt.legend()

    # 保存图像
    plt.savefig("training_plots/rmse_k.png")
    plt.close()  # 关闭图像

    print("Plot saved as 'training_plots/rmse_k.png'.")
