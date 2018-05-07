import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from visual_im.utils.dm_env import EnvSpec
from torch.utils.data import TensorDataset, DataLoader

class MLPBaseline:
    def __init__(self, env_spec: EnvSpec, hidden_sizes=(64,64), learning_rate=1e-5, epoch=10, batch=10, seed=None):
        self.feature_size = env_spec.observation_dim + 4
        self.loss_fn = nn.MSELoss(size_average=False)
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        self.epoch = epoch
        # torch.manual_seed(seed)
        self.batch = batch
        self.model = nn.Sequential()
        self.model.add_module('fc_0', nn.Linear(self.feature_size, self.hidden_sizes[0]))
        self.model.add_module('tanh_0', nn.Tanh())
        self.model.add_module('fc_1', nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]))
        self.model.add_module('tanh_1', nn.Tanh())
        self.model.add_module('fc_2', nn.Linear(self.hidden_sizes[1], 1))

    def _features(self, path):
        # compute regression features for the path
        o = np.clip(path["observations"], -10, 10)
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 1000.0
        feat = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)
        return feat

    def fit(self, paths, return_errors=False):

        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        dataset = TensorDataset(torch.FloatTensor(featmat), torch.FloatTensor(returns))
        data_loader = DataLoader(dataset, batch_size=self.batch, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if return_errors:
            error_before = self.get_error(data_loader)

        for _ in range(self.epoch):
            for batch_idx, (data, target) in enumerate(data_loader):
                data = Variable(data)
                target = Variable(target).float()
                predictions = self.model(data)
                loss = self.loss_fn(predictions, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if return_errors:
            error_after = self.get_error(data_loader)
            return error_before/np.sum(returns**2), error_after/np.sum(returns**2)

    def get_error(self, data_loader):
        error = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = Variable(data)
            target = Variable(target).float()
            predictions = self.model(data)
            error += self.loss_fn(predictions, target)
        return error

    def predict(self, path):
        if self.model is None:
            return np.zeros(len(path["rewards"]))
        return self.model(Variable(torch.FloatTensor(self._features(path)))).data.numpy().reshape(-1)
