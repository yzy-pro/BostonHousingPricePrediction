import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas
import torch
import matplotlib.pyplot

# read the datas
train_data = pandas.read_csv(
    'house-prices-advanced-regression-techniques/train.csv')
test_data = pandas.read_csv(
    'house-prices-advanced-regression-techniques/test.csv')
processed_datas = pandas.concat(
    (train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# standardize the data
numeric_features = (
    processed_datas.dtypes[processed_datas.dtypes != 'object'].index)
processed_datas[numeric_features] = processed_datas[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(0)
# OHE
processed_datas = pandas.get_dummies(processed_datas, dummy_na=True)

# csv2tensor
processed_datas = processed_datas.astype('float32')
n_train = train_data.shape[0]
processed_train_datas = torch.tensor(processed_datas[:n_train].values,
                                     dtype=torch.float32)
processed_test_datas = torch.tensor(processed_datas[n_train:].values,
                                    dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                            dtype=torch.float32)

# MLP
class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_units=256, num_hidden_layers=3):
        super(MLP, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(
            torch.nn.Linear(in_features, hidden_units))

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

# assessment
loss_fn = torch.nn.MSELoss()
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), min=1.0)
    rmse = torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

# train
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_loss, test_loss = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            loss = loss_fn(net(X), y)
            loss.backward()
            optimizer.step()

        train_loss.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_loss.append(log_rmse(net, test_features, test_labels))

        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}'
                f', Train RMSE: {train_loss[-1]:.6f}')

    return train_loss, test_loss
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, num_hidden_layers):
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = MLP(in_features=X_train.shape[1], num_hidden_layers=num_hidden_layers)
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate,
                                       weight_decay, batch_size)#*data
        train_loss_sum += train_loss[-1]
        valid_loss_sum += valid_loss[-1]

        # matplotlib.pyplot.plot(range(1, num_epochs + 1), train_loss,
        #                        label='train')
        # matplotlib.pyplot.plot(range(1, num_epochs + 1), valid_loss,
        #                        label='valid')
        # matplotlib.pyplot.xlabel('epoch')
        # matplotlib.pyplot.ylabel('rmse')
        # matplotlib.pyplot.legend()
        # matplotlib.pyplot.yscale('log')
        # matplotlib.pyplot.show()
        #
        # print(
        #     f'fold{i + 1}，train log rmse{train_loss[-1]:f}, valid log rms'
        #     f'e{valid_loss[-1]:f}')

    return train_loss_sum / k, valid_loss_sum / k

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size, num_hidden_layers):
    net = MLP(in_features=train_features.shape[1]
              , num_hidden_layers=num_hidden_layers
              , hidden_units=hidden_units)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)

    matplotlib.pyplot.plot(range(1, num_epochs + 1), train_ls)
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.ylabel('log rmse')
    matplotlib.pyplot.xlim([1, num_epochs])
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.show()

    print(f'train log rmse：{train_ls[-1]:f}')

    predictions = net(test_features).detach().numpy()
    test_data['SalePrice'] = pandas.Series(predictions.reshape(1, -1)[0])
    submission = pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    return submission

## settings!!!
(k, num_epochs, learing_rate, weight_decay, batch_size, num_hidden_layers ,
 hidden_units)= \
    (5, 1500, 0.00005, 1e-4, 128, 3, 256)

train_l, valid_l = k_fold(k, processed_train_datas, train_labels, num_epochs,
                          learing_rate, weight_decay, batch_size, num_hidden_layers)
print(f'{k}-fold: average train log rmse: {train_l:f}, '
      f'average valid log rmse: {valid_l:f}')

submission = train_and_pred(processed_train_datas, processed_test_datas,
                      train_labels,
               test_data, num_epochs, learing_rate, weight_decay, batch_size, num_hidden_layers)
submission.to_csv('submission.csv', index=False)