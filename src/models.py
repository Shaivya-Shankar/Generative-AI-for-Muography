import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple, Optional, Union, Any, Callable, Dict

import zuko

import lightning as L
from pytorch_lightning.utilities import move_data_to_device

import torchmetrics

import functools
import numpy as np

from enum import Enum, auto
from abc import ABCMeta, abstractmethod

import matplotlib
import matplotlib.pyplot as plt


###################################################################################
#                              Abstract class                                     #
###################################################################################

class AbsNetwork(L.LightningModule,metaclass=ABCMeta):
    def __init__(
        self,
        optimizer: optim.Optimizer = None,
        scheduler_config: dict[str,Any] = None,
        metrics: dict[str,Any] = {},
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.metrics = nn.ModuleDict(metrics)

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def set_scheduler_config(self,scheduler_config):
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):
        if self.optimizer is None:
            raise RuntimeError('Optimizer not set')
        if self.scheduler_config is None:
            return self.optimizer
        else:
            return {
                'optimizer' : self.optimizer,
                'lr_scheduler': self.scheduler_config,
            }

    @abstractmethod
    def _loss(self,y,t):
        pass

    @property
    def loss_weighting(self):
        return None

    def loss(self,y,t):
        # Get model depend loss #
        loss_values = self._loss(y,t)
        # Make total loss if several are provided #
        if torch.is_tensor(loss_values):
            loss_values = {'loss_tot':loss_values}
        elif isinstance(loss_values,dict):
            assert 'loss_tot' not in loss_values.keys(),f'loss_tot is a reserved key for loss dict'
            if self.loss_weighting is None:
                loss_values['loss_tot'] = sum(loss_values.values())
            else:
                assert len(set(self.loss_weighting.keys()).intersection(set(loss_values.keys()))) == len(loss_values.keys()), f'Mismatch between losses {loss_values.keys()} and weighting {self.loss_weighting.keys()} keys'
                loss_values['loss_tot'] = sum([self.loss_weighting[k] * loss_values[k] for k in loss_values.keys()])
        else:
            raise RuntimeError(f'Type {type(loss_values)} of loss values unknown')
        # Safety checks #
        for key,values in loss_values.items():
            if torch.isnan(values).any() or torch.isinf(values).any():
                message = f'There are {(torch.isnan(values)*1).sum()} nan and {(torch.isinf(values)*1).sum()} inf values in {key}  (out of {values.shape[0]} entries)'
                raise RuntimeError(message)
        # Average all losses now (after we got total loss) #
        loss_values = {
            key: values.mean()
            for key,values in loss_values.items()
        }
        return loss_values


    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def log_values_after_validation(self):
        return None

    def shared_eval(self, batch, batch_idx, prefix):
        t = batch['target']
        y = self(batch)
        # Compute losses and log them #
        losses = self.loss(y,t)
        for key,loss in losses.items():
            self.log(f"{prefix}/{key}", loss, prog_bar=True)
        if self.metrics is not None:
            for metric_name,metric in self.metrics.items():
                metric(y.ravel(),t.ravel())
                self.log(f"{prefix}/{metric_name}", metric, prog_bar=True)
        if prefix == 'val':
            log_values = self.log_values_after_validation()
            if log_values is not None:
                assert isinstance(log_values,dict)
                for key,val in log_values.items():
                    self.log(key, val, prog_bar=False, on_step=False, on_epoch=True)

        return losses['loss_tot']

def preprocess(f):
    @functools.wraps(f)
    def wrapper(self, x, *args):
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-9)
        return f(self,x,*args)
    return wrapper

###################################################################################
#                           Deep neural network                                   #
###################################################################################
class DeepNeuralNetwork(AbsNetwork):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation_first: bool = False,
        neurons: List[int] = [],
        dropout: float = 0.,
        loss_function: Optional[nn.Module] = None,
        **kwargs,
    ):
        # Call abs class #
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore='loss_function')

        # Save attributes #
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.activation_first = activation_first
        self.neurons = neurons
        self.dropout = dropout
        self.loss_function = loss_function

        # Initialize layers #
        self.layers = []
        for i in range(len(self.neurons)+1):
            in_neurons = self.neurons[i - 1] if i > 0 else self.dim_in
            out_neurons = self.neurons[i] if i < len(self.neurons) else self.dim_out
            self.layers.append(nn.Linear(in_neurons, out_neurons))
            if i < len(self.neurons):
                if self.hidden_activation is not None and activation_first:
                    self.layers.append(self.hidden_activation())
                if self.layer_norm:
                    self.layers.append(nn.LayerNorm(out_neurons))
                if self.batch_norm:
                    self.layers.append(nn.BatchNorm1d(out_neurons))
                if self.hidden_activation is not None and not activation_first:
                    self.layers.append(self.hidden_activation())
            else:
                if self.output_activation is not None:
                    self.layers.append(self.output_activation())
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*self.layers)

    #def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    def forward(self, batch):
        return self.layers(batch['input'])

    def _loss(self,y,t):
        if self.loss_function is not None:
            return self.loss_function(y,t)
        else:
            raise NotImplementedError('No loss function defined')

###################################################################################
#                           Adversarial network                                   #
###################################################################################

class GAN(AbsNetwork):
    def __init__(self, G, D, lam):
        super().__init__()
        self.G = G
        self.D = D
        self.lam = lam

    def loss(self,x,y,z,w):
        return self.G.loss(x,y,w) - self.lam * self.D.loss(self.G(x),z,w)

    def forward(self, x):
        return self.D(self.G(x))

    def freeze_G(self):
        for p in self.G.parameters():
            p.requires_grad = False

    def freeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = False

    def unfreeze_G(self):
        for p in self.G.parameters():
            p.requires_grad = True

    def unfreeze_D(self):
        for p in self.D.parameters():
            p.requires_grad = True


###################################################################################
#                         Variational Auto-Encoder                                #
###################################################################################

class VariationalAutoencoder(AbsNetwork):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_cond: int,
        dim_latent: int,
        dim_embed: Optional[int] = None,
        loss_function: Optional[nn.Module] = None,
        neurons_embed: List[int] = [],
        neurons_encoder: List[int] = [],
        neurons_decoder: List[int] = [],
        hidden_activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        batch_norm: bool = False,
        layer_norm: bool = False,
        variational: bool = False,
        dropout: float = 0.,
        beta: float = 1.,
        free_bits: float = 0.,
        eps: float = 1e-15,
        **kwargs
    ):
        #self.save_hyperparameters(ignore=['backbone'])
        super().__init__(**kwargs)

        if len(self.metrics) > 0:
            print ('Metrics not available in MDN, will disable them')
            self.metrics = {}

        # Save attributes #
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_cond = dim_cond
        self.dim_embed = dim_embed
        self.dim_latent = dim_latent
        self.neurons_embed = neurons_embed
        self.neurons_encoder = neurons_encoder
        self.neurons_decoder = neurons_decoder
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.variational = variational
        self.dropout = dropout
        self.beta = beta
        self.free_bits = free_bits
        self.loss_function = loss_function

        # Register epsilon value #
        self.register_buffer('eps',torch.tensor(eps))

        # Make encoder and decoder #
        if self.dim_embed:
            self.cond_embedding = DeepNeuralNetwork(
                dim_in = self.dim_cond,
                dim_out = self.dim_embed,
                neurons = self.neurons_embed,
                hidden_activation = self.hidden_activation,
                output_activation = None,
                batch_norm = self.batch_norm,
                layer_norm = self.layer_norm,
                dropout = self.dropout,
            )
            dim_encoder = self.dim_in + self.dim_embed
            dim_decoder = self.dim_latent + self.dim_embed
        else:
            self.cond_embedding = None
            dim_encoder = self.dim_in + self.dim_cond
            dim_decoder = self.dim_latent + self.dim_cond

        self.encoder = DeepNeuralNetwork(
            dim_in = dim_encoder,
            dim_out = 2 * self.dim_latent if self.variational else self.dim_latent,
            neurons = self.neurons_encoder,
            hidden_activation = self.hidden_activation,
            output_activation = None,
            batch_norm = self.batch_norm,
            layer_norm = self.layer_norm,
            dropout = self.dropout,
        )
        self.decoder = DeepNeuralNetwork(
            dim_in = dim_decoder,
            dim_out = self.dim_out,
            neurons = self.neurons_decoder,
            hidden_activation = self.hidden_activation,
            output_activation = self.output_activation,
            batch_norm = self.batch_norm,
            layer_norm = self.layer_norm,
            dropout = self.dropout,
        )

    def encode(self,x,c):
        xc = torch.cat((x,c),dim=-1)
        h = self.encoder({'input':xc})
        return h

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(mu).to(mu.device)
        return mu + eps * sigma

    def decode(self,z,c):
        zc = torch.cat((z,c),dim=-1)
        return self.decoder({'input':zc})

    def forward(self,batch: Dict[str,Tensor]):
        # Recover elements #
        x = batch['input']
        c = batch['condition']
        if self.cond_embedding is not None:
            c = self.cond_embedding({'input':c})

        # Encode #
        h = self.encode(x,c)

        # Reparameterize #
        if self.variational:
            mu = h[...,:self.dim_latent]
            sigma = F.softplus(h[...,self.dim_latent:]) + self.eps
            z = self.reparameterize(mu,sigma)
        else:
            z = h

        # Decode #
        y = self.decode(z,c)

        if self.variational:
            return y, mu, sigma
        else:
            return y

    def loss_kl(self,mu,sigma):
        # KL(q||p) = -0.5 * sum(1 + logσ² - μ² - σ²)
        kl_per_dim = -0.5 * (1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2))
        kl_per_dim = torch.clamp(kl_per_dim, min = self.free_bits)
        return kl_per_dim.sum(dim=-1)

    @property
    def loss_weighting(self):
        return {'loss_reco' : 1., 'loss_kl': self.beta}

    def log_values_after_validation(self):
        return {'beta': self.beta}

    def _loss(self, y, t):
        if self.variational:
            assert len(y) == 3
            y, mu, sigma = y

        losses = {'loss_reco' : self.loss_function(y,t).sum(dim=-1)}
        if self.variational:
            losses['loss_kl'] = self.loss_kl(mu,sigma)
        return losses

    def sample(self,batch,N=1):
        if not self.variational:
            raise RuntimeError(f'Model needs to have `variational = True` to be able to sample')
        batch = move_data_to_device(batch,self.device)
        c = batch['condition']
        if self.cond_embedding is not None:
            c = self.cond_embedding({'input':c})
        normal = torch.distributions.Normal(
            torch.zeros(
                (c.shape[0],N,self.dim_latent),
                 device=c.device,
            ),
            torch.ones(
                (c.shape[0],N,self.dim_latent),
                 device=c.device,
            ),
        )
        c = c.unsqueeze(dim=1).repeat_interleave(N,dim=1)
        z = normal.sample()
        with torch.no_grad():
            y = self.decode(
                z.reshape(
                    z.shape[0] * N,
                    z.shape[2],
                ),
                c.reshape(
                    c.shape[0] * N,
                    c.shape[2],
                ),
            )
        return y.reshape(
            y.shape[0]//N,
            N,
            y.shape[1],
        ).permute((0,2,1)) # B,N,D




###################################################################################
#                         Mixture density network                                 #
###################################################################################

class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()

#class MixtureDensityNetwork(AbsNetwork):
#    """
#    Mixture density network.
#    [ Bishop, 1994 ]
#    Heavily inspired from
#    https://github.com/tonyduan/mixture-density-network
#    Parameters
#    ----------
#    dim_in: int; dimensionality of the covariates
#    dim_out: int; dimensionality of the response variable
#    n_components: int; number of components in the mixture model
#    """
#
#    def __init__(
#        self,
#        backbone,
#        dim_out,
#        n_components,
#        neurons = [],
#        noise_type=NoiseType.DIAGONAL,
#        fixed_noise_level=None,
#        eps=1e-15,
#        **kwargs
#    ):
#        super().__init__(**kwargs)
#        if len(self.metrics) > 0:
#            self.metrics = {}
#
#        # Save attributes #
#        self.backbone = backbone
#        self.dim_in = self.backbone.dim_out
#        self.dim_out = dim_out
#        self.n_components = n_components
#        self.neurons = neurons
#        self.noise_type = noise_type
#        self.fixed_noise_level = fixed_noise_level
#
#        self.save_hyperparameters(ignore=['backbone'])
#
#        # Determine number of channels #
#        if self.noise_type.value == NoiseType.DIAGONAL.value:
#            self.num_sigma_channels = self.dim_out * n_components
#        elif self.noise_type.value == NoiseType.ISOTROPIC.value:
#            self.num_sigma_channels = n_components
#        elif self.noise_type.value == NoiseType.ISOTROPIC_ACROSS_CLUSTERS.value:
#            self.num_sigma_channels = 1
#        elif self.noise_type.value == NoiseType.FIXED.value:
#            self.num_sigma_channels = 0
#        else:
#            # Sometimes when with jupyter reload, weird things happen with the enum check
#            # Safe by value here
#            raise NotImplementedError
#        print (f'Will use the {self.noise_type} noise type')
#
#        # Register epsilon value #
#        self.register_buffer('eps',torch.tensor(eps))
#
#        # Get the end of backbone parameters #
#        base_args = {
#            'dim_in': self.dim_in,
#            'neurons': self.neurons,
#        }
#        if isinstance(self.backbone,DeepNeuralNetwork):
#            base_args.update(
#                {
#                    'hidden_activation': self.backbone.hidden_activation,
#                    'batch_norm': self.backbone.batch_norm,
#                    'layer_norm': self.backbone.layer_norm,
#                    'dropout': self.backbone.dropout,
#                    'neurons': self.backbone.neurons[::-1],
#                }
#            )
#        else:
#            raise NotImplementedError
#
#        # Make final branches #
#        self.pi = DeepNeuralNetwork(
#            output_activation = None,
#            dim_out = self.n_components,
#            **base_args,
#        )
#        self.mu = DeepNeuralNetwork(
#            output_activation = None,
#            dim_out = self.dim_out * self.n_components,
#            **base_args,
#        )
#        if self.num_sigma_channels > 0:
#            self.sigma = DeepNeuralNetwork(
#                output_activation = None,
#                dim_out = self.num_sigma_channels,
#                **base_args,
#            )
#
#
#    def forward(self,x,c):
#        # Make intermediate state #
#        h = self.backbone(x)
#        # Get each output #
#        mu    = self.mu(h)
#        sigma = self.sigma(h) if self.noise_type.value != NoiseType.FIXED.value else None
#        pi    = self.pi(h)
#        # Rescale #
#        #sigma = torch.exp(sigma)
#        sigma = F.elu(sigma) + 1 + self.eps
#        pi    = F.softmax(pi,dim=-1)
#        # Process sigma #
#        if self.noise_type.value == NoiseType.DIAGONAL.value:
#            sigma = sigma + self.eps
#        elif self.noise_type.value == NoiseType.ISOTROPIC.value:
#            sigma = (sigma + self.eps).repeat(1, self.dim_out)
#        elif self.noise_type.value == NoiseType.ISOTROPIC_ACROSS_CLUSTERS.value:
#            sigma = (sigma + self.eps).repeat(1, self.n_components * self.dim_out)
#        elif self.noise_type.value == NoiseType.FIXED.value:
#            sigma = torch.log(torch.full_like(mu, fill_value=self.fixed_noise_level))
#        else:
#            raise RuntimeError('Noise type {self.noise_type} not understood')
#        # Reshapes #
#        mu = mu.reshape(-1, self.n_components, self.dim_out)
#        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
#        if torch.isinf(sigma).any() or torch.isnan(sigma).any():
#            raise RuntimeError(f'There are {((torch.isinf(sigma))*1).sum()} inf and {((torch.isnan(sigma))*1).sum()} nan entries for log(sigma) (out of {len(sigma.ravel())}), this should not happen')
#
#
#        return pi, mu, sigma
#
#    def log_likelihood(self,y,t):
#        pi, mu, sigma = y
#        z = (t.unsqueeze(1) - mu) / sigma
#        return torch.log(pi) \
#            - 0.5 * torch.einsum("bij,bij->bi", z, z) \
#            - torch.sum(torch.log(sigma),dim=-1)
#
#    def _loss(self, y, t):
#        return - torch.logsumexp(self.log_likelihood(y,t),dim=-1)
#
#    def sample(self, x, c, N=1):
#        # Forward #
#        with torch.no_grad():
#            pi, mu, sigma = self.forward(x, c)
#        z_shape = mu.shape + (N,)
#        z = torch.normal(mean=torch.zeros(z_shape), std=torch.ones(z_shape))
#        pi = pi.unsqueeze(-1) # adapt pi shape to mu, sigma
#        y = torch.sum(pi[...,None] * (z * sigma[...,None] + mu[...,None]), dim=1)
#        # Need to add the sample dimension
#        return y


class MixtureDensityNetwork(AbsNetwork):
    """
    Mixture density network.
    [ Bishop, 1994 ]
    Heavily inspired from
    https://github.com/tonyduan/mixture-density-network (pytorch)
    https://github.com/CoteDave/blog/blob/master/Made%20easy/MDN%20regression/mdn_model.py (tf)
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim_out: int,
        n_components: int,
        neurons: List[int] = [],
        noise_type: NoiseType = NoiseType.DIAGONAL,
        fixed_noise_level: Optional[float] =  None,

        eps=1e-15,
        **kwargs
    ):
        self.save_hyperparameters(ignore=['backbone'])
        super().__init__(**kwargs)

        if len(self.metrics) > 0:
            print ('Metrics not available in MDN, will disable them')
            self.metrics = {}

        # Save attributes #
        self.backbone = backbone
        self.dim_in = self.backbone.dim_out
        self.dim_out = dim_out
        self.n_components = n_components
        self.neurons = neurons
        self.noise_type = noise_type
        self.fixed_noise_level = fixed_noise_level


        # Determine number of channels #
        if self.noise_type.value == NoiseType.DIAGONAL.value:
            self.num_sigma_channels = self.dim_out * n_components
        elif self.noise_type.value == NoiseType.ISOTROPIC.value:
            self.num_sigma_channels = n_components
        elif self.noise_type.value == NoiseType.ISOTROPIC_ACROSS_CLUSTERS.value:
            self.num_sigma_channels = 1
        elif self.noise_type.value == NoiseType.FIXED.value:
            self.num_sigma_channels = 0
        else:
            # Sometimes when with jupyter reload, weird things happen with the enum check
            # Safe by value here
            raise NotImplementedError
        print (f'Will use the {self.noise_type} noise type')

        # Register epsilon value #
        self.register_buffer('eps',torch.tensor(eps))

        # Get the end of backbone parameters #
        base_args = {
            'dim_in': self.dim_in,
            'neurons': self.neurons,
        }
        base_args.update(
            {
                'hidden_activation': self.backbone.hidden_activation,
                #'batch_norm': self.backbone.batch_norm,
                #'layer_norm': self.backbone.layer_norm,
                #'dropout': self.backbone.dropout,
            }
        )

        # Make final branches #
        self.pi = DeepNeuralNetwork(
            output_activation = None,
            dim_out = self.n_components,
            **base_args,
        )
        self.mu = DeepNeuralNetwork(
            output_activation = None,
            dim_out = self.dim_out * self.n_components,
            **base_args,
        )
        if self.num_sigma_channels > 0:
            self.sigma = DeepNeuralNetwork(
                output_activation = None,
                dim_out = self.num_sigma_channels,
                **base_args,
            )


    def forward(self,batch: Dict[str,Tensor]):
        # Make intermediate state #
        h = self.backbone(batch)
        # Get each output #
        mu    = self.mu({'input':h})
        sigma = self.sigma({'input':h}) if self.noise_type.value != NoiseType.FIXED.value else None
        pi    = self.pi({'input':h})
        # Rescale #
        sigma = F.softplus(sigma)
        log_pi = F.log_softmax(pi,dim=-1)
        # Process sigma #
        if self.noise_type.value == NoiseType.DIAGONAL.value:
            sigma = sigma + self.eps
        elif self.noise_type.value == NoiseType.ISOTROPIC.value:
            sigma = (sigma + self.eps).repeat(1, self.dim_out)
        elif self.noise_type.value == NoiseType.ISOTROPIC_ACROSS_CLUSTERS.value:
            sigma = (sigma + self.eps).repeat(1, self.n_components * self.dim_out)
        elif self.noise_type.value == NoiseType.FIXED.value:
            sigma = torch.log(torch.full_like(mu, fill_value=self.fixed_noise_level))
        else:
            raise RuntimeError('Noise type {self.noise_type} not understood')
        # Reshapes #
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        if torch.isinf(sigma).any() or torch.isnan(sigma).any():
            raise RuntimeError(f'There are {((torch.isinf(sigma))*1).sum()} inf and {((torch.isnan(sigma))*1).sum()} nan entries for sigma (out of {len(sigma.ravel())}), this should not happen')


        return log_pi, mu, sigma

    def log_likelihood(self,y,t):
        log_pi, mu, sigma = y
        # log_pi : [B,M]
        # mu     : [B,M,D]
        # sigma  : [B,M,D]
        # t (target) has shape [B,D]
        # B = batch size, M = number of gaussians, D = output dimension
        t = t.unsqueeze(1).expand_as(mu) # [B,D] -> [B,M,D]
        normal = torch.distributions.normal.Normal(loc=mu,scale=sigma)
        log_probs = normal.log_prob(t) # [B,M,D]
        log_probs = log_probs.sum(dim=2) # [B,M]
        log_probs += log_pi
        log_probs = torch.logsumexp(log_probs, dim=1) # [B]
        return log_probs

    def log_likelihood_gaussian(self,y,t):
        log_pi, mu, sigma = y
        z = (t.unsqueeze(1) - mu) / sigma
        return torch.logsumexp(
            log_pi \
            - 0.5 * torch.einsum("bij,bij->bi", z, z) \
            - torch.sum(torch.log(sigma),dim=-1),
            axis = -1,
        )

    def _loss(self, y, t):
        return -self.log_likelihood(y,t)
        # return - self.log_likelihood_gaussian(y,t)

    def sample_gaussian_once(self, x, c):
        # Forward #
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(x, c)
        cum_pi = torch.cumsum(torch.exp(pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples

    def sample_gaussian(self, x, c, N):
        # Forward #
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(x, c)
        log_pi = log_pi.repeat_interleave(repeats=N,dim=0)
        mu = mu.repeat_interleave(repeats=N,dim=0)
        sigma = sigma.repeat_interleave(repeats=N,dim=0)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x)*N, 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples.reshape(x.shape[0],N,mu.shape[-1]).permute(0,2,1)

    def sample(self,batch,N=1):
        batch = move_data_to_device(batch,self.device)
        # Forward #
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(batch)
        # log_pi : [B,M]
        # mu     : [B,M,D]
        # sigma  : [B,M,D]
        # need to return [B,D,N] (N=sample)
        B,M,D = mu.shape
        log_pi = log_pi.unsqueeze(1).expand(B, N, M).reshape(-1, M)       # [B*N, M]
        mu     = mu.unsqueeze(1).expand(B, N, M, D).reshape(-1, M, D)     # [B*N, M, D]
        sigma  = sigma.unsqueeze(1).expand(B, N, M, D).reshape(-1, M, D)  # [B*N, M, D]

        # Sample from log_pi and select mu and sigma #
        cat = torch.distributions.categorical.Categorical(logits=log_pi)
        indices = cat.sample()
        sel_mu = mu[torch.arange(B * N), indices]       # [B*N, D]
        sel_sigma = sigma[torch.arange(B * N), indices] # [B*N, D]

        # Use these mu and sigma to sample the proper gaussians #
        normal = torch.distributions.Normal(sel_mu, sel_sigma)
        sample = normal.sample() # [B*N, D]
        return sample.reshape(B,N,D).permute((0,2,1)) # returns [B,D,N]

    def sample_once(self,x,c):
        # Forward #
        with torch.no_grad():
            log_pi, mu, sigma = self.forward(x, c)
        rand_normal = torch.distributions.normal.Normal(loc=mu,scale=sigma).sample()
        probs = torch.distributions.categorical.Categorical(logits=log_pi).sample().long()
        probs = probs.unsqueeze(-1).repeat_interleave(self.dim_out,dim=-1)
        samples = torch.take_along_dim(rand_normal, indices=probs.unsqueeze(1), dim=1).squeeze(1)
        return samples


###################################################################################
#                                Normalising Flow                                 #
###################################################################################

class NormalisingFlow(AbsNetwork):
    def __init__(
        self,
        dim_in: int,
        dim_cond: int,
        bins: int,
        transforms: int,
        dim_embed: Optional[int] = None,
        neurons_embed: List[int] = [],
        randperm: bool = True,
        passes: Optional[int] = 2,
        neurons: List[int] = [64,64],
        hidden_activation: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Save attributes #
        self.dim_in = dim_in
        self.dim_cond = dim_cond
        self.dim_embed = dim_embed
        self.neurons_embed = neurons_embed
        self.bins = bins
        self.transforms = transforms
        self.randperm = randperm
        self.passes = passes
        self.neurons = neurons
        self.hidden_activation = hidden_activation

        # Condition embeddding #
        if self.dim_embed is not None:
            self.cond_embedding = DeepNeuralNetwork(
                dim_in = self.dim_cond,
                dim_out = self.dim_embed,
                neurons = self.neurons_embed,
                hidden_activation = self.hidden_activation,
                output_activation = None,
            )
            dim_context = self.dim_embed
        else:
            self.cond_embedding = None
            dim_context = self.dim_cond

        # Make flow #
        self.flow = zuko.flows.NSF(
            features = self.dim_in,
            context = dim_context,
            bins = self.bins,
            transforms = self.transforms,
            randperm = self.randperm,
            passes = self.passes,
            hidden_features = self.neurons,
            activation = self.hidden_activation
        )

    def forward(self,batch):
        if self.cond_embedding is None:
            context = batch['condition']
        else:
            context = self.cond_embedding({'input': batch['condition']})
        return - self.flow(context).log_prob(batch['target'])

    def _loss(self, y, t):
        # y is the -log prob
        return y

    def sample(self, batch, N=1):
        batch = move_data_to_device(batch,self.device)
        if self.cond_embedding is None:
            context = batch['condition']
        else:
            context = self.cond_embedding({'input': batch['condition']})
        with torch.no_grad():
            samples = self.flow(context).sample((N,))
        return samples.permute(1,2,0) # [B,D,N]  

