import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决在使用 PyTorch 时，可能会遇到的 OpenMP 相关问题

import pandas  # 用于数据处理
import torch   # 用于深度学习
import matplotlib.pyplot  # 用于绘制图形

# 读取数据
train_data = pandas.read_csv(
    'house-prices-advanced-regression-techniques/train.csv')  # 训练数据集
test_data = pandas.read_csv(
    'house-prices-advanced-regression-techniques/test.csv')  # 测试数据集
processed_datas = pandas.concat(
    (train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # 合并训练集和测试集（去除 ID 列和目标标签列）

# 标准化数据
numeric_features = (
    processed_datas.dtypes[processed_datas.dtypes != 'object'].index)  # 获取所有数值型特征的列名
processed_datas[numeric_features] = processed_datas[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))  # 对数值特征进行标准化处理
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(0)  # 填充缺失值为 0

# 独热编码（One-Hot Encoding，OHE）处理类别特征
processed_datas = pandas.get_dummies(processed_datas, dummy_na=True)  # 对所有类别变量进行 OHE，dummy_na=True 处理缺失值

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
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), min=1.0)  # 对预测值做裁剪，确保预测值不小于 1
    rmse = torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels)))  # 对数 RMSE
    return rmse.item()

# 训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_loss, test_loss = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)  # 数据加载器，按 batch_size 从训练数据中随机采样
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # Adam 优化器

    for epoch in range(num_epochs):
        for X, y in train_iter:  # 对每个 batch 进行训练
            optimizer.zero_grad()  # 清除梯度
            loss = loss_fn(net(X), y)  # 计算当前 batch 的损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

        train_loss.append(log_rmse(net, train_features, train_labels))  # 计算并记录训练集的 RMSE
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels))  # 计算并记录验证集的 RMSE

        if (epoch + 1) % 100 == 0:  # 每 100 个 epoch 输出一次训练信息
            print(
                f'Epoch {epoch + 1}/{num_epochs}'
                f', Train RMSE: {train_loss[-1]:.6f}')

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

# 执行 K 折交叉验证
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, num_hidden_layers):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = MLP(in_features=X_train.shape[1], num_hidden_layers=num_hidden_layers)
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate,
                                       weight_decay, batch_size)  # 训练模型并获取损失值
        train_loss_sum += train_loss[-1]  # 累加训练损失
        valid_loss_sum += valid_loss[-1]  # 累加验证损失

    return train_loss_sum / k, valid_loss_sum / k  # 返回 K 折的平均训练损失和验证损失

# 训练并进行预测
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size, num_hidden_layers):
    net = MLP(in_features=train_features.shape[1],
              num_hidden_layers=num_hidden_layers,
              hidden_units=hidden_units)  # 初始化模型
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)  # 训练模型

    # 绘制训练损失曲线
    matplotlib.pyplot.plot(range(1, num_epochs + 1), train_ls)
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.ylabel('log rmse')
    matplotlib.pyplot.xlim([1, num_epochs])
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.show()

    print(f'train log rmse：{train_ls[-1]:f}')

    # 进行预测
    predictions = net(test_features).detach().numpy()  # 用训练好的模型进行预测
    test_data['SalePrice'] = pandas.Series(predictions.reshape(1, -1)[0])  # 将预测值保存到 test_data 中
    submission = pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)  # 创建提交结果

    onnx_filename = "boston_housing_model.onnx"
    dummy_input = torch.randn(1, train_features.shape[1])  # Match input
    # dimensions
    torch.onnx.export(net, dummy_input, onnx_filename, export_params=True,
                      opset_version=11)

    return submission

## 超参数设置
(k, num_epochs, learing_rate, weight_decay, batch_size, num_hidden_layers ,
 hidden_units)= \
    (2, 100, 0.005, 1e-4, 128, 3, 256)

# 执行 K 折交叉验证
train_l, valid_l = k_fold(k, processed_train_datas, train_labels, num_epochs,
                          learing_rate, weight_decay, batch_size, num_hidden_layers)
print(f'{k}-fold: average train log rmse: {train_l:f}, '
      f'average valid log rmse: {valid_l:f}')

# 训练并生成预测结果
submission = train_and_pred(processed_train_datas, processed_test_datas,
                      train_labels,
               test_data, num_epochs, learing_rate, weight_decay, batch_size, num_hidden_layers)

# 将预测结果保存到 CSV 文件
submission.to_csv('submission.csv', index=False)


