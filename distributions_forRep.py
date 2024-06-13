import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import bernoulli, normal
import numpy as np
from torch import optim
import torch.distributions
class FullyConnected(nn.Sequential):
    """
    FullyConnected network
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """
    @staticmethod
    def get_class(dtype):
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))

class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])    ##   ### 指明输出维度是2

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return normal.Normal(loc, scale)
    
class DiagNormalNet(nn.Module):
    """
        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale
    


class p_x_zi_zc_za_ze(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, zc, zt, zy, ze):
        z_concat = torch.cat((zc, zt, zy, ze), -1)
        loc_scale = self.fc(z_concat)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        x = normal.Normal(loc, scale)
        return x

class p_xbin_zi_zc_za_ze(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])   #

    def forward(self, zc, zt, zy, ze):
        z_concat = torch.cat((zc, zt, zy, ze), -1)
        logits = self.fc(z_concat).squeeze(-1).clamp(min=0, max=11)
        x_bin = bernoulli.Bernoulli(logits=logits)
        return x_bin


class p_t_zc_zt(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, zc,zt):
        z_concat = torch.cat((zc, zt), -1)
        loc_scale = self.fc(z_concat)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        t =  normal.Normal(loc, scale)
        return t

class p_y_t_zc_zy(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, t, zc, zy):
        z_concat = torch.cat((zc, zy, t.unsqueeze(1)), -1)
        z_concat = z_concat.float()
        loc_scale = self.fc(z_concat)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        y = normal.Normal(loc, scale)
        return y


class q_z_x(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_layers,latent_dim):
        super().__init__()
        self.z_nn = FullyConnected([x_dim] +
                                   [hidden_dim] * (num_layers - 1),
                                   final_activation=nn.ELU())
        self.z_out_nn = DiagNormalNet([hidden_dim, latent_dim])

    def forward(self, x):
        hidden = self.z_nn(x.float())
        params = self.z_out_nn(hidden)
        # z = normal.Normal(*params)
        return params
    
class q_t_zc_zt(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, zc,zt):
        input_concat = torch.cat((zc, zt), -1)
        loc_scale = self.fc(input_concat)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        t = normal.Normal(loc, scale)
        return t

class q_y_t_zc_zy(nn.Module):
    def __init__(self, latent_dim_c, latent_dim_y, hidden_dim, num_layers):
        super().__init__()
        self.y_nn = FullyConnected([latent_dim_c + latent_dim_y + 1 ] +
                                   [hidden_dim] * (num_layers - 1),
                                   final_activation=nn.ELU())
        self.y_out_nn = NormalNet([hidden_dim])

    def forward(self, t,zc,zy):
        x = torch.cat((zc, zy, t.unsqueeze(1)), -1)
        hidden = self.y_nn(x.float())
        params = self.y_out_nn(hidden)
        y = normal.Normal(*params)
        return y
    
def init_qz(q_z_x_dist, p_z_dist, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)
    
    optimizer = optim.Adam(q_z_x_dist.parameters(), lr=0.001)
    
    for i in range(50):
        batch = np.random.choice(idx, 1)
        
        x_train= torch.cuda.FloatTensor(x[batch])
        
        z_infer = q_z_x_dist(x_train)
        
        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean ** 2 - 1)).sum(1)
        
        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        
        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')
    
    return q_z_x_dist
