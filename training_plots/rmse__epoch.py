import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置环境变量，解决在使用 PyTorch 时，可能会遇到的 OpenMP 相关问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 读取数据
train_data = pd.read_csv(
    '../house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv(
    '../house-prices-advanced-regression-techniques/test.csv')
processed_datas = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理：标准化数值特征并填充缺失值
numeric_features = processed_datas.dtypes[
    processed_datas.dtypes != 'object'].index
processed_datas[numeric_features] = processed_datas[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(0)

# 独热编码处理类别特征
processed_datas = pd.get_dummies(processed_datas, dummy_na=True)

# 将数据转换为 Tensor 类型
processed_datas = processed_datas.astype('float32')
n_train = train_data.shape[0]
processed_train_datas = torch.tensor(processed_datas[:n_train].values,
                                     dtype=torch.float32)
processed_test_datas = torch.tensor(processed_datas[n_train:].values,
                                    dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                            dtype=torch.float32)


# 定义多层感知机模型
class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_units=256, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(in_features, hidden_units))

        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_units, hidden_units))

        self.output = torch.nn.Linear(hidden_units, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x


# 损失函数：均方误差（MSE）
loss_fn = torch.nn.MSELoss()


# 计算训练日志 RMSE（均方根误差）
def log_rmse(net, features, labels, device):
    features = features.to(device)
    labels = labels.to(device)
    clipped_preds = torch.clamp(net(features), min=1.0)
    rmse = torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 训练函数
def train(net, train_features, train_labels, valid_features, valid_labels,
          num_epochs, learning_rate, weight_decay, batch_size, device):
    train_loss, valid_loss = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        if epoch % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}')
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()

        # 记录每个 epoch 的训练和验证损失
        train_loss.append(log_rmse(net, train_features, train_labels, device))
        valid_loss.append(log_rmse(net, valid_features, valid_labels, device))

    return train_loss, valid_loss


# 主函数
if __name__ == "__main__":
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置隐藏层数量为 3
    num_hidden_layers = 3
    train_rmses = []
    valid_rmses = []

    # 拆分训练数据为训练集和验证集
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        processed_train_datas, train_labels, test_size=0.2, random_state=42)

    # 创建并训练模型
    print(f"Running with num_hidden_layers={num_hidden_layers}")
    net = MLP(in_features=processed_train_datas.shape[1],
              num_hidden_layers=num_hidden_layers).to(device)
    train_loss, valid_loss = train(net, train_features, train_labels,
                                   valid_features, valid_labels,
                                   num_epochs=1000,
                                   learning_rate=0.00005, weight_decay=1e-3,
                                   batch_size=128, device=device)

    # 绘制损失随 epoch 变化的图
    os.makedirs("training_plots", exist_ok=True)
    plt.plot(range(1, 1001), train_loss, label='Training Loss')
    plt.plot(range(1, 1001), valid_loss, label='Validation Loss',
             linestyle='--')
    plt.ylim(0, 1)
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE vs Epochs')
    plt.legend()

    # 保存图像
    plt.savefig("training_plots/rmse_vs_epoch.png")
    plt.close()

    print("Plot saved as 'training_plots/rmse_vs_epoch.png'.")
