import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy.io as sio


def plot_rate(rate_his, rolling_intv=50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)  # rate_his transformed to array type
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color='b', alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()


class MemoryDNN:
    def __init__(self, net, learning_rate=0.05, training_interval=10, batch_size=100, memory_size=1000):
        self.net = net
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all allocation strategies
        self.enumerate_actions = []

        # stored memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),  # 设置网络的全连接层
            nn.LeakyReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.LeakyReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every multiple steps
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])  # 一种包含单一数据类型元素的多维矩阵
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])

        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=0.0001)
        criterion = nn.MSELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert (self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h):
        # to have batch dimension when feed into Tensor
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()

        return m_pred[0]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
        # plot_rate(self.cost_his, 5, 'Training Loss')
        sio.savemat('./Loss.mat', {'x': np.arange(len(self.cost_his)) * self.training_interval, 'Loss': self.cost_his})
