import os
import math
import yaml
import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import lightning as L
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader

from utils import predict

EPS = 1e-12

class HistoryCallback(Callback):
    """
    https://github.com/Lightning-AI/pytorch-lightning/discussions/16258
    """
    def __init__(
        self,
        config,
        groups,
        plot_columns=None,
        axes_params=None,
        logs=None,
        dirpath=None,
        filename = 'history.yml',
    ):
        super().__init__()
        self.config = config
        self.logs = {c: [] for c in config.keys()} if logs is None else logs
        self.groups = groups
        self.plot_columns = min(len(self.groups),plot_columns) if plot_columns is not None else len(self.groups)
        self.axes_params = axes_params
        self.dirpath = dirpath
        self.filename = filename
        if self.axes_params is not None:
            assert isinstance(self.axes_params,list)
            assert len(self.axes_params) == len(self.groups), f'{self.axes_params} has length {len(self.axes_params)} but there are {len(self.groups)} groups {self.groups}'


    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:  # optional skip
            return
        # Add to logs #
        for key in self.logs.keys():
            if key not in trainer.callback_metrics.keys():
                raise RuntimeError(f'Could not find {key} in trainer metric {trainer.callback_metrics.keys()}')
            self.logs[key].append(trainer.callback_metrics[key].cpu().item())

        # Get figure and save #
        fig = self.plot(show=False)
        if isinstance(trainer.logger,L.pytorch.loggers.comet.CometLogger):
            trainer.logger.experiment.log_figure(
                figure_name = 'history',
                figure = fig,
                overwrite = True,
                step = trainer.current_epoch,
            )
        plt.close(fig)
        # Log the content #
        if self.dirpath is not None:
            self.save(os.path.join(self.dirpath,self.filename))

    def plot(self,show=False):
        n_cols = self.plot_columns
        n_rows = math.ceil(len(self.groups)/self.plot_columns)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_cols == 1 and n_rows == 1:
            axs = [axs]
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

        for idx, names in enumerate(self.groups):
            draw_legend = False
            i_col = idx % n_cols
            i_row = idx // n_cols
            if n_rows > 1:
                ax = axs[i_row, i_col]
            else:
                ax = axs[i_col]
            if not isinstance(names, tuple) and not isinstance(names, list):
                names = [names]
            for name in names:
                ax.plot(torch.arange(len(self.logs[name]))+1, self.logs[name], **self.config[name])
                if "label" in self.config[name].keys():
                    draw_legend = True
                if self.axes_params is not None:
                    for pname,pval in self.axes_params[idx].items():
                        getattr(ax,pname)(pval)
                if show:
                    print (f'... {name:15s} : {self.logs[name][-1]:10.5f}')
            ax.set_xlabel("Epochs")
            if draw_legend:
                ax.legend(loc="upper right")

        if show:
            plt.show()
        return fig

    def save(self,path):
        config = {
            'config' : self.config,
            'groups' : self.groups,
            'axes_params' : self.axes_params,
            'plot_columns' : self.plot_columns,
            'logs' : self.logs,
        }
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    @classmethod
    def load(cls,path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        return cls(**config)

class KLAnnealingCallback(Callback):
    def __init__(self,annealing):
        self.annealing = annealing

    def on_train_epoch_start(self, trainer, pl_module):
        beta = self.annealing(trainer.current_epoch)
        pl_module.beta = beta


class BaseCallback(Callback):
    def on_validation_epoch_end(self,trainer,pl_module):
        if trainer.sanity_checking:
            return
        if trainer.current_epoch % self.frequency != 0 or trainer.current_epoch == 0:
            return

        # Get figures #
        figs = self.make_plots(pl_module,disable_tqdm=True)

        for figname,fig in figs.items():
            trainer.logger.experiment.log_figure(
                figure_name = figname,
                figure = fig,
                overwrite = True,
                step = trainer.current_epoch,
            )
            plt.close(fig)

class BaseSamplingCallback(BaseCallback):
    def __init__(
        self,
        values,
        scaling,
        plotting_config,
        preprocessed = False,
        frequency = 1,
        bins = 51,
        N_sample = 1000,
        suffix = "",
    ):
        self.values = values
        self.scaling = scaling
        self.plotting_config = plotting_config
        self.frequency = frequency
        self.preprocessed = preprocessed
        self.bins = bins
        self.N_sample = N_sample
        self.suffix = suffix

        self._cond_indices = torch.cartesian_prod(
            *[
                torch.arange(cond.shape[0])
                for cond in self.values
            ]
        )
        self._cond_bins = [
            self._compute_bins_from_centers(cond)
            for cond in self.values
        ]
        self.batch = self.scale_batch({self._batch_name:torch.cartesian_prod(*self.values)})

    def _compute_bins_from_centers(self,values):
        if len(values) == 1:
            val = values[0]
            bin_edges = np.array([val-0.5,val+0.5])
        else:
            midpoints = 0.5 * (values[1:] + values[:-1])
            first_edge = values[0] - 0.5 * (values[1] - values[0])
            last_edge = values[-1] + 0.5 * (values[-1] - values[-2])
            bin_edges = np.concatenate([[first_edge], midpoints, [last_edge]])
        return bin_edges

    def scale_batch(self,batch):
        return {
            key: self.scaling.transform(
                name = key,
                x = tensor,
                features = self.plotting_config[key]['features'],
            )
            for key,tensor in batch.items()
        }

    def process(self,model):
        # Sample and forward (need preprocessing before) #
        model.eval()
        with torch.no_grad():
            batch = move_data_to_device(self.batch,model.device)
            samples = model.sample(batch,N=self.N_sample).cpu()
            if self._batch_name == 'input':
                y = model(self.batch).cpu()
            else:
                y = None
        return y,samples

    def make_plots(self,model,show=False,**kwargs):
        y,samples = self.process(model)
        # Undo samples preprocessing #
        if self.preprocessed:
            pass
        else:
            samples = self.scaling.inverse(
                name = 'target',
                x = samples.permute(
                    (0,2,1)
                ).reshape(
                    samples.shape[0] * self.N_sample,
                    samples.shape[1],
                ),
                features = self.plotting_config['target']['features'],
            ).reshape(
                samples.shape[0],
                self.N_sample,
                samples.shape[1],
            ).permute((0,2,1))

        figs = self.plot_sampling(y,samples)
        if show:
            plt.show()
        return figs

    def make_binning(self,samples):
        bins = []
        for i in range(samples.shape[1]):
            if not self.preprocessed and self.plotting_config['target']['logscales'][i]:
                bins.append(
                    np.logspace(
                        math.log10(samples[:,i,:].min()),
                        math.log10(samples[:,i,:].max()),
                        self.bins,
                    )
                )
            else:
                bins.append(
                    np.linspace(
                        samples[:,i,:].min(),
                        samples[:,i,:].max(),
                        self.bins,
                    )
                )
        return bins

    def plot_1D(self,samples):
        # samples shape : [N,Do,S]
        Di = len(self.values)
        Do = samples.shape[1]
        fig,axs = plt.subplots(nrows=Di,ncols=Do,figsize=(5*Do,4*Di))
        if len(self.suffix) > 0:
            plt.suptitle(f'Material : {self.suffix}',fontsize=16)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        bins = self.make_binning(samples)
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        if axs.ndim == 1:
            axs = axs.reshape(-1,1)
        for i in range(Di):
            N = self.values[i].shape[0]
            colors = plt.cm.plasma(np.linspace(0, 1, N))
            for o in range(Do):
                for j in range(N):
                    axs[i,o].hist(
                        samples[self._cond_indices[:,i]==j,o,:].ravel(),
                        bins = bins[o],
                        color = colors[j],
                        histtype = 'step',
                        density = True,
                    )
                h = axs[i,o].hist(
                    samples[:,o,:].ravel(),
                    bins = bins[o],
                    color = 'black',
                    linestyle = '--',
                    histtype = 'step',
                    density = True,
                )
                axs[i,o].set_xlabel(self.plotting_config['target']['labels'][o],fontsize=14)
                axs[i,o].set_yscale('log')
                if not self.preprocessed and self.plotting_config['target']['logscales'][o]:
                    axs[i,o].set_xscale('log')
                    axs[i,o].set_ylim(h[0][h[0]>0].min(),None)
                else:
                    axs[i,o].set_ylim(0,None)
        return fig

    def plot_2D(self,samples):
        Di = len(self.values)
        Do = samples.shape[1]
        fig,axs = plt.subplots(nrows=Di,ncols=Do,figsize=(5*Do,4*Di))
        if len(self.suffix) > 0:
            plt.suptitle(f'Material : {self.suffix}',fontsize=16)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        bins = self.make_binning(samples)
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        if axs.ndim == 1:
            axs = axs.reshape(-1,1)
        for i in range(Di):
            N = self.values[i].shape[0]
            colors = plt.cm.plasma(np.linspace(0, 1, N))
            for o in range(Do):
                array = np.zeros((N,len(bins[o])-1))
                for j in range(N):
                    array[j,:] = np.histogram(
                        samples[self._cond_indices[:,i]==j,o,:].ravel(),
                        bins = bins[o],
                    )[0]
                h = axs[i,o].pcolormesh(
                    self._cond_bins[i],
                    bins[o],
                    array.T,
                    cmap = 'viridis',
                    norm = matplotlib.colors.LogNorm(vmin=1e-1 if array.sum() > 0 else None),
                )
                if not self.preprocessed and self.plotting_config['target']['logscales'][o]:
                    axs[i,o].set_yscale('log')
                if self.plotting_config[self._batch_name]['logscales'][i]:
                    axs[i,o].set_xscale('log')
                plt.colorbar(h,ax=axs[i,o])
                axs[i,o].set_xlabel(self.plotting_config[self._batch_name]['labels'][i],fontsize=14)
                axs[i,o].set_ylabel(self.plotting_config['target']['labels'][o],fontsize=14)
        return fig

    def plot_pair(self,samples):
        D = samples.shape[1]
        fig,axs = plt.subplots(nrows=D,ncols=D,figsize=(5*D,4*D))
        if len(self.suffix) > 0:
            plt.suptitle(f'Material : {self.suffix}',fontsize=16)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        bins = self.make_binning(samples)
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        for i in range(D):
            for j in range(D):
                if i < j :
                    axs[i,j].set_axis_off()
                elif i == j:
                    axs[i,j].hist(
                        samples[:,i,:].ravel(),
                        bins = bins[i],
                    )
                    if not self.preprocessed and self.plotting_config['target']['logscales'][i]:
                        axs[i,j].set_xscale('log')
                    axs[i,j].set_yscale('log')
                    axs[i,j].set_ylim(1e-1,None)
                    axs[i,j].set_xlabel(self.plotting_config['target']['labels'][i],fontsize=14)
                else:
                    h = axs[i,j].hist2d(
                        samples[:,i,:].ravel(),
                        samples[:,j,:].ravel(),
                        bins = (bins[i],bins[j]),
                        norm = matplotlib.colors.LogNorm(vmin=1e-1),
                    )
                    if not self.preprocessed and self.plotting_config['target']['logscales'][i]:
                        axs[i,j].set_xscale('log')
                    if not self.preprocessed and self.plotting_config['target']['logscales'][j]:
                        axs[i,j].set_yscale('log')
                    plt.colorbar(h[3],ax=axs[i,j])
                    axs[i,j].set_xlabel(self.plotting_config['target']['labels'][i],fontsize=14)
                    axs[i,j].set_ylabel(self.plotting_config['target']['labels'][j],fontsize=14)
        return fig


class VAESamplingCallback(BaseSamplingCallback):
    _batch_name = 'condition'

    def plot_sampling(self,y,samples):
        return {
            f'sampling_1D_{self.suffix}' : self.plot_1D(samples),
            f'sampling_2D_{self.suffix}' : self.plot_2D(samples),
            f'sampling_pair_{self.suffix}' : self.plot_pair(samples),
        }

class NFSamplingCallback(BaseSamplingCallback):
    _batch_name = 'condition'

    def plot_sampling(self,y,samples):
        return {
            f'sampling_1D_{self.suffix}' : self.plot_1D(samples),
            f'sampling_2D_{self.suffix}' : self.plot_2D(samples),
            f'sampling_pair_{self.suffix}' : self.plot_pair(samples),
        }

# --- NEW CLASS FOR GAN ---
class GANSamplingCallback(BaseSamplingCallback):
    _batch_name = 'condition'

    def process(self, model):
        # For a GAN, the 'model' is the generator.
        generator = model
        generator.eval()
        
        # Get the device from the model's parameters, which is a robust way
        device = next(generator.parameters()).device
        
        with torch.no_grad():
            # Get the batch of conditions and move to the correct device
            batch = move_data_to_device(self.batch, device)
            conditions = batch[self._batch_name]
            
            # Generate N_sample samples for each condition
            all_samples = []
            for _ in range(self.N_sample):
                noise = torch.randn(conditions.size(0), generator.noise_dim, device=device)
                samples = generator(conditions, noise).cpu()
                all_samples.append(samples.unsqueeze(2))
            
            # Concatenate along the new sample dimension
            samples = torch.cat(all_samples, dim=2)
            
        # A GAN doesn't have a direct forward pass like a VAE, so y is None
        return None, samples

    def plot_sampling(self, y, samples):
        return {
            f'sampling_1D_{self.suffix}': self.plot_1D(samples),
            f'sampling_2D_{self.suffix}': self.plot_2D(samples),
            f'sampling_pair_{self.suffix}': self.plot_pair(samples),
        }
# --- END OF NEW CLASS ---


class MDNSamplingCallback(BaseSamplingCallback):
    _batch_name = 'input'

    def plot_parameter(self,param):
        Di = len(self.values)
        C = param.shape[1]
        if param.dim() == 2:
            param = param.unsqueeze(-1)
        Do = param.shape[2]
        fig,axs = plt.subplots(nrows=Di,ncols=Do,figsize=(5*Do,4*Di))
        bins = np.linspace(param.min(),param.max(),self.bins)
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        if axs.ndim == 1:
            axs = axs.reshape(-1,1)
        colors = plt.cm.rainbow(np.linspace(0, 1, C))
        for i in range(Di):
            N = self.values[i].shape[0]
            for o in range(Do):
                for c in range(C):
                    min_p = []
                    max_p = []
                    mean_p = []
                    for j in range(N):
                        p = param[self._cond_indices[:,i]==j,c,o]
                        min_p.append(p.min().item())
                        max_p.append(p.max().item())
                        mean_p.append(p.mean().item())
                    print (i,o,min_p,max_p,mean_p)
                    axs[i,o].plot(
                        self.values[i],
                        mean_p,
                        color = colors[c],
                    )
                axs[i,o].set_xlabel(self.plotting_config[self._batch_name]['labels'][i],fontsize=14)
                axs[i,o].set_ylabel(self.plotting_config['target']['labels'][o],fontsize=14)
        return fig


    def plot_sampling(self,y,samples):
        # Get MDN parameters #
        log_pi, mu, sigma = y
        pi = torch.exp(log_pi)

        return {
            f'sampling_1D_{self.suffix}' : self.plot_1D(samples),
            f'sampling_2D_{self.suffix}' : self.plot_2D(samples),
            f'sampling_pair_{self.suffix}' : self.plot_pair(samples),
        }


class BasePredictionCallback(BaseCallback):
    def __init__(
        self,
        dataset,
        preprocessed = False,
        frequency = 1,
        bins = 51,
        batch_size = 1024,
        N_batch = math.inf,
        suffix = "",
    ):
        self.dataset = dataset
        self.loader = DataLoader(self.dataset,batch_size=batch_size,shuffle=False)
        self.preprocessed = preprocessed
        self.frequency = frequency
        self.bins = bins
        self.N_batch = N_batch
        self.suffix = suffix
        self.targets = self.dataset.tensors['target']
        self.inputs = self.dataset.tensors['input']

    def make_plots(self,model,disable_tqdm=False,show=False):
        preds = predict(
            model = model,
            data = self.loader,
            N_batch = self.N_batch,
            disable_tqdm = disable_tqdm,
        )
        if isinstance(preds,(list,tuple)):
            assert torch.is_tensor(preds[0])
            N = preds[0].shape[0]
        elif torch.is_tensor(preds):
            N = preds.shape[0]
        else:
            raise ValueError(f'Preds type {type(preds)} not understood')
        figs = self.plot_preds(preds)
        if show:
            plt.show()
        return figs

    def make_binning(self,y,t):
        bins = []
        for i in range(y.shape[-1]):
            if not self.preprocessed and self.dataset.plotting_config['target']['logscales'][i]:
                bins.append(
                    np.logspace(
                        math.log10(min(y[...,i].min(),t[...,i].min())),
                        math.log10(max(y[...,i].max(),t[...,i].max())),
                        self.bins,
                    )
                )
            else:
                bins.append(
                    np.linspace(
                        min(y[...,i].min(),t[...,i].min()),
                        max(y[...,i].max(),t[...,i].max()),
                        self.bins,
                    )
                )
        return bins

    def plot_1D(self,y,t):
        D = y.shape[-1]
        fig,axs = plt.subplots(nrows=1,ncols=D,figsize=(5*D,4))
        if len(self.suffix) > 0:
            plt.suptitle(f'Material : {self.suffix}',fontsize=16)
        bins = self.make_binning(y,t)
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        for i in range(D):
            axs[i].hist(
                y[:,i],
                bins = bins[i],
                color = 'royalblue',
                histtype = 'step',
                label = 'Prediction',
            )
            axs[i].hist(
                t[:,i],
                bins = bins[i],
                color = 'orange',
                histtype = 'step',
                label = 'Truth',
            )
            axs[i].legend(fontsize=14)
            axs[i].set_yscale('log')
            axs[i].set_xlabel(self.dataset.plotting_config['target']['labels'][i],fontsize=14)
            if not self.preprocessed and self.dataset.plotting_config['target']['logscales'][i]:
                axs[i].set_xscale('log')
        return fig

    def plot_2D(self,y,t):
        D = y.shape[-1]
        fig,axs = plt.subplots(nrows=1,ncols=D,figsize=(5*D,4))
        if len(self.suffix) > 0:
            plt.suptitle(f'Material : {self.suffix}',fontsize=16)
        bins = self.make_binning(y,t)
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        for i in range(D):
            h = axs[i].hist2d(
                t[:,i],
                y[:,i],
                bins = bins[i],
                norm = matplotlib.colors.LogNorm(vmin=1e-1),
            )
            if not self.preprocessed and self.dataset.plotting_config['target']['logscales'][i]:
                axs[i].set_xscale('log')
                axs[i].set_yscale('log')
            plt.colorbar(h[3],ax=axs[i])
            axs[i].set_xlabel(
                '{} (Truth)'.format(self.dataset.plotting_config['target']['labels'][i]),
                fontsize = 14,
            )
            axs[i].set_ylabel(
                '{} (Prediction)'.format(self.dataset.plotting_config['target']['labels'][i]),
                fontsize = 14,
            )
        return fig



class VAEPredictionCallback(BasePredictionCallback):
    def plot_preds(self,preds):
        if torch.is_tensor(preds):
            y = preds
        elif isinstance(preds,(tuple,list)):
            y,mu,sigma = preds
        else:
            raise NotImplementedError
        t = self.targets[:y.shape[0]]
        if not self.preprocessed:
            y = self.dataset.scaling.inverse(
                name = 'target',
                x = y,
                features = self.dataset.plotting_config['target']['features'],
            )
            t = self.dataset.scaling.inverse(
                name = 'target',
                x = t,
                features = self.dataset.plotting_config['target']['features'],
            )

        figs = {
            f'preds_1D_{self.suffix}' : self.plot_1D(y,t),
            f'preds_2D_{self.suffix}' : self.plot_2D(y,t),
        }

        if isinstance(preds,(tuple,list)):
            fig,axs = plt.subplots(ncols=2,figsize=(12,5))
            if len(self.suffix) > 0:
                plt.suptitle(f'Material : {self.suffix}',fontsize=16)
            bins_mu = np.linspace(mu.min(),mu.max(),self.bins)
            for i in range(mu.shape[-1]):
                axs[0].hist(mu[:,i],bins=bins_mu,histtype='step')
            bins_sigma = np.linspace(sigma.min(),sigma.max(),self.bins)
            for i in range(sigma.shape[-1]):
                axs[1].hist(sigma[:,i],bins=bins_sigma,histtype='step')
            axs[0].set_yscale('log')
            axs[0].set_xlabel('$\mu$',fontsize=14)
            axs[1].set_yscale('log')
            axs[1].set_xlabel('$\sigma$',fontsize=14)
            figs[f'parameters_{self.suffix}'] = fig

        return figs

class ClassifierPredictionCallback(BasePredictionCallback):
    def plot_preds(self,preds):
        y = preds
        t = self.targets[:y.shape[0]]
        x = self.inputs[:y.shape[0]]
        if not self.preprocessed:
            y = self.dataset.scaling.inverse(
                name = 'target',
                x = y,
                features = self.dataset.plotting_config['target']['features'],
            )
            t = self.dataset.scaling.inverse(
                name = 'target',
                x = t,
                features = self.dataset.plotting_config['target']['features'],
            )
            x = self.dataset.scaling.inverse(
                name = 'input',
                x = x,
                features = self.dataset.plotting_config['input']['features'],
            )

        N = len(self.dataset.plotting_config['input']['features'])
        fig = plt.figure(figsize=(6*N,5))
        gs = GridSpec(1, N, width_ratios=[1]*N, wspace=0.4, bottom=0.3)
        for i in range(N):
            gs_sub = GridSpecFromSubplotSpec(
                nrows = 2,
                ncols = 1,
                subplot_spec = gs[i],
                height_ratios = [1.0,0.2],
                hspace = 0.1,
            )
            ax1 = fig.add_subplot(gs_sub[0,0])
            ax2 = fig.add_subplot(gs_sub[1,0])
            if not self.preprocessed and self.dataset.plotting_config['input']['logscales'][i]:
                bins = np.logspace(math.log10(x[:,i].min()),math.log10(x[:,i].max()),self.bins)
            else:
                bins = np.linspace(x[:,i].min(),x[:,i].max(),self.bins)
            full_content = np.histogram(x[:,i],bins=bins)[0]
            true_content = np.histogram(x[:,i][t.ravel()>0],bins=bins)[0]
            pred_content = np.histogram(x[:,i],bins=bins,weights=y.ravel())[0]

            true_var = np.sqrt(true_content) / (full_content+EPS)
            true_content = true_content / (full_content + EPS)
            pred_content = pred_content / (full_content + EPS)

            true_up = true_content + true_var
            true_do = true_content - true_var

            ratio = pred_content / (true_content + EPS)
            ratio_up = (true_content + true_var) / (true_content + EPS)
            ratio_do = (true_content - true_var) / (true_content + EPS)


            ax1.stairs(
                values = true_content,
                edges = bins,
                color = 'orange',
                linewidth = 1,
                label = 'Truth',
            )
            ax1.fill_between(
                x = bins,
                y1 = np.r_[true_do,true_do[-1]],
                y2 = np.r_[true_up,true_up[-1]],
                color = 'orange',
                alpha = 0.3,
                step = 'post',
            )
            ax1.stairs(
                values = pred_content,
                edges = bins,
                color = 'royalblue',
                linewidth = 1,
                label = 'Predicted',
            )
            ax2.stairs(
                values = ratio,
                edges = bins,
                color = 'royalblue',
                linewidth = 1,
                label = 'Predicted',
            )
            ax2.fill_between(
                x = bins,
                y1 = np.r_[ratio_do,ratio_do[-1]],
                y2 = np.r_[ratio_up,ratio_up[-1]],
                color = 'orange',
                alpha = 0.3,
                step = 'post',
            )

            if not self.preprocessed and self.dataset.plotting_config['input']['logscales'][i]:
                ax1.set_xscale('log')
                ax2.set_xscale('log')
            ax1.set_xticklabels([])
            ax1.legend(fontsize=12)
            ax1.set_ylabel(r'$P(\Delta r = X_0 / 100)$')
            ax2.set_ylim(0.5,1.5)
            ax2.set_xlabel(self.dataset.plotting_config['input']['labels'][i],fontsize=14)
            ax2.set_ylabel(r'$\frac{Predicted}{Truth}$')
            ax2.grid(visible=True,which='major',axis='y')


        return {
            'classifier' : fig,
        }
