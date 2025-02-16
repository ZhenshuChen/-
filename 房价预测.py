import os
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 数据预览
print(f"训练数据形状: {train_data.shape}")
print(f"测试数据形状: {test_data.shape}")

# 处理特征
processed_datas = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 确保所有特征都是数值类型，处理无法转换的字段
processed_datas = processed_datas.apply(pd.to_numeric, errors='coerce')

# 标准化数值特征
numeric_features = processed_datas.dtypes[processed_datas.dtypes != 'object'].index
processed_datas[numeric_features] = (processed_datas[numeric_features].apply
                                     (lambda x: (x - x.mean()) / (x.std())))
processed_datas[numeric_features] = processed_datas[numeric_features].fillna(0)

# 独热编码
processed_datas = pd.get_dummies(processed_datas, dummy_na=True)

# 转换为张量
n_train = train_data.shape[0]
processed_train_datas = torch.tensor(processed_datas[:n_train].values,dtype=torch.float32)      #特征张量
processed_test_datas = torch.tensor(processed_datas[n_train:].values,dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),dtype=torch.float32)     #标签张量


# 定义新的多层感知机模型 (3层）
class MLP(nn.Module):
    def __init__(self, in_features, hidden_units=256):
        super(MLP, self).__init__()     #调用父类 nn.Module 的构造函数，确保正确初始化模型
        # 隐藏层
        self.hidden1 = nn.Linear(in_features, hidden_units)
        self.hidden2 = nn.Linear(hidden_units, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# 定义损失函数
loss_fn = nn.MSELoss()


# 定义log_rmse
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), min=1.0)
    rmse = torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


# 训练函数 (adam算法）
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = (torch.utils.data.DataLoader
                  (torch.utils.data.TensorDataset
                   (train_features, train_labels),batch_size, shuffle=True))
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss_fn(net(X), y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))

        # 输出每个epoch的训练损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train RMSE: {train_ls[-1]:.6f}')

    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1        #保证k＞1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)     #计算折的索引范围
        X_part, y_part = X[idx, :], y[idx]      #提取折的数据
        if j == i:      #检验是否为当前折
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:       #拼接训练集
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = MLP(in_features=X_train.shape[1])  # 使用新的 MLP 模型
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == k - 1:  # 最后一个折才显示图表
            plt.plot(range(1, num_epochs + 1), train_ls, label='train')
            plt.plot(range(1, num_epochs + 1), valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.legend()
            plt.yscale('log')
            plt.show()

        print(f'折{i + 1}，训练log rmse{train_ls[-1]:f}, 验证log rmse{valid_ls[-1]:f}')

    return train_l_sum / k, valid_l_sum / k


# 训练并提交
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = MLP(in_features=train_features.shape[1])  # 使用新的 MLP 模型
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    plt.plot(range(1, num_epochs + 1), train_ls)
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.xlim([1, num_epochs])
    plt.yscale('log')
    plt.show()

    print(f'训练log rmse：{train_ls[-1]:f}')

    # 将网络应用于测试集
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

# 超参数设置
k, num_epochs, lr, weight_decay, batch_size = 5, 1800, 0.0005, 1e-5, 128
train_l, valid_l = k_fold(k, processed_train_datas, train_labels, num_epochs,lr,weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {train_l:f}, 'f'平均验证log rmse: {valid_l:f}')

# 训练并预测提交
train_and_pred(processed_train_datas, processed_test_datas, train_labels,test_data,num_epochs, lr, weight_decay, batch_size)