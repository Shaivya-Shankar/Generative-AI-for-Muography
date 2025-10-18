import matplotlib
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import awkward as ak
from tqdm.notebook import tqdm
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from functools import cached_property
from typing import Optional, List
from numpy.typing import NDArray
from numba import jit

class HistSampler:
    def __init__(
        self,
        data: NDArray,
        bins: int,
        label: str,
        interpolate_sampling: bool = True,
        interpolate_cdf_factor: Optional[int] = None,
    ):
        # Save attributes #
        self.N_data = data.shape[0]
        self.bins = bins
        self.label = label
        self.interpolate_sampling = interpolate_sampling
        self.interpolate_cdf_factor = interpolate_cdf_factor
        # Safety checks #
        if data.ndim == 1:
            data = data.reshape(-1,1)
        else:
            assert data.shape[1] == 1
        # Get PDF #
        self.pdf,self.grid = np.histogram(data,self.bins)
        self.pdf = self.pdf.astype(self.grid.dtype)
        if self.pdf.sum() != data.shape[0]:
            print (f'Binning for {self.label} : missing {data.shape[0]-self.pdf.sum()} steps [{abs(data.shape[0]-self.pdf.sum())/data.shape[0]*100:6.3f}%]')
        # Normalize PDF #
        self.widths = np.diff(self.grid)
        self.pdf /= self.widths
        integral = (self.pdf*self.widths).sum()
        self.pdf /= integral
        # Get CDF #
        self.cdf = np.cumsum(self.pdf * self.widths)

        #self.cdf /= self.cdf[-1]
        #self.cdf[0] = 0.
        # Linear interpolation #
        if self.interpolate_cdf_factor is not None:
            assert isinstance(self.interpolate_cdf_factor, int)
            if self.interpolate_cdf_factor <= 1:
                raise RuntimeError(f'An interpolation factor <=1 will just decrease CDF precision')
            spline = interp1d(
                x = self.grid[1:],
                y = self.cdf,
                kind = 'quadratic',
                bounds_error = False,
                fill_value = 0, #'extrapolate',
            )
            self.grid_int = np.unique(
                np.array(
                    [
                        np.r_[edge_left:edge_right:complex(self.interpolate_cdf_factor+1)]
                        for edge_left,edge_right in zip(self.grid[:-1],self.grid[1:])
                    ]
                )
            )
            self.cdf_int = spline(self.grid_int[1:])
            self.cdf_int[self.cdf_int<0] = 0.
            self.cdf_int[self.cdf_int>1] = 1.
            # correct out-of-bounds correction (set to 0) on the right end to 1
            self.cdf_int[self.cdf_int.argmax():] = 1.

            self.pdf_int = np.r_[0,np.diff(self.cdf_int)]
            self.widths_int = np.diff(self.grid_int)
            self.pdf_int /= self.widths_int
            integral = (self.pdf_int*self.widths_int).sum()
            self.pdf_int /= integral

    @staticmethod
    @jit(nopython=True)
    def _sample(grid: NDArray, cdf: NDArray, N: int, interpolate:bool = False) -> NDArray:
        y = np.random.rand(N)
        centers = (grid[1:]+grid[:-1])/2
        idx = np.searchsorted(cdf, y, side='right')
        idx = np.clip(idx, 1, len(grid) - 2)  # prevent out-of-bounds
        if interpolate:
            xa = grid[idx]
            xb = grid[idx+1]
            ya = cdf[idx-1]
            yb = cdf[idx]
            x = (y-ya)/(yb-ya) * (xb-xa) + xa
        else:
            x = centers[idx]
        return x

    def sample(self, N: int):
        if self.interpolate_cdf_factor is not None:
            return self._sample(self.grid_int,self.cdf_int,N,self.interpolate_sampling)
        else:
            return self._sample(self.grid,self.cdf,N,self.interpolate_sampling)

    def plot_sampling(self,ax=None,N=None,bins=None,xscale='linear',yscale='linear',title=None,batch_size=None,show=False):
        # Make figure #
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(6,5))
        else:
            fig = None
        # Make sample binning #
        if bins is None:
            bins = self.grid
        elif isinstance(bins,int):
            if xscale == 'linear':
                bins = np.linspace(self.grid.min(),self.grid.max(),bins)
            elif xscale == 'log':
                bins = np.logspace(np.log10(self.grid.min()),np.log10(self.grid.max()),bins)
            else:
                raise NotImplementedError
        else:
            raise RuntimeError('Bins should be an int')
        # Get sampled numbers #
        if N is None:
            N = self.N_data
        if batch_size is None:
            sampled = np.histogram(self.sample(N),bins,density=True)[0]
        else:
            N_batches = math.ceil(N/batch_size)
            counts = np.zeros(len(bins)-1)
            for i in tqdm(range(N_batches),desc='Sampling'):
                n_batch = int(batch_size) if i < N_batches-1 else int(N_batches % batch_size)
                if n_batch == 0:
                    continue
                counts += np.histogram(
                    self.sample(n_batch),
                    bins = bins,
                    density = False,
                )[0]
            sampled = counts / (sum(counts) * np.diff(bins))
        # Plot #
        ax.stairs(
            values = self.pdf,
            edges = self.grid,
            fill = False,
            color = 'blue',
            label = 'PDF',
        )
        ax.stairs(
            values = sampled,
            edges = bins,
            fill = False,
            color = 'orange',
            label = 'Sampling',
        )
        ax.set_xlabel(self.label,fontsize=14)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if yscale == 'linear':
            ax.set_ylim(0,None)
        if yscale == 'log':
            ax.set_ylim(1e-1,None)
        if yscale == 'linear':
            ax.set_ylim(0,max(self.pdf.max(),sampled.max())*1.2)
        if yscale == 'log':
            ax.set_ylim(
                min(1e-2,self.pdf[self.pdf>0].min()/2),
                max(self.pdf.max(),sampled.max()) * 2,
            )

        ax.legend()
        if show:
            plt.show()
        if fig is not None:
            if title is not None:
                plt.suptitle(title,fontsize=16)
            return fig


    def plot_distribution(self,axs=None,xscale='linear',yscale='linear',show=False):
        if axs is not None:
            assert len(axs) == 2
            fig = None
        else:
            fig,axs = plt.subplots(ncols=2,figsize=(12,5))
        # PDF #
        axs[0].stairs(
            values = self.pdf,
            edges = self.grid,
            fill = False,
            color = 'blue',
            label = 'Original'
        )
        axs[0].set_title('PDF')
        axs[0].set_xlabel('x')
        axs[0].set_xscale(xscale)
        axs[0].set_yscale(yscale)
        if yscale == 'linear':
            axs[0].set_ylim(0,self.pdf.max()*1.2)
        if yscale == 'log':
            axs[0].set_ylim(
                min(1e-2,self.pdf[self.pdf>0].min()/2),
                self.pdf.max() * 2,
            )

        # CDF #
        axs[1].stairs(
            values = self.cdf,
            edges = self.grid,
            fill = False,
            color = 'blue',
            label = 'Original'
        )
        if self.interpolate_cdf_factor is not None:
            axs[0].stairs(
                values = self.pdf_int,
                edges = self.grid_int,
                fill = False,
                color = 'orange',
                label = 'Interpolated'
            )
            axs[0].legend()
            axs[1].stairs(
                values = self.cdf_int,
                edges = self.grid_int,
                fill = False,
                color = 'orange',
                label = 'Interpolated'
            )
            axs[1].legend()
        axs[1].set_title('CDF')
        axs[1].set_xlabel('x')
        axs[1].set_xscale(xscale)
        axs[1].set_yscale(yscale)
        if yscale == 'linear':
            axs[1].set_ylim(0,1)
        if yscale == 'log':
            axs[1].set_ylim(
                min(1e-2,self.cdf[self.cdf>0].min()/2),
                1.,
            )

        if show:
            plt.show()
        if fig is not None:
            return fig

class MultiVarSampler:
    def __init__(
        self,
        samplers: List[HistSampler],
    ):
        assert len(samplers) == 4
        self.samplers = samplers

    def plot_sampling(self,title=None,**kwargs):
        fig,axs = plt.subplots(ncols=4,figsize=(20,5))
        plt.suptitle(title,fontsize=16)
        plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0.2)
        show = kwargs.pop('show',False)
        xscale = kwargs.pop('xscale',False)
        for i in range(len(self.samplers)):
            self.samplers[i].plot_sampling(
                ax = axs[i],
                show = False,
                xscale = xscale if i != 2 else 'linear',
                **kwargs,
            )
        if show:
            plt.show()
        return fig

    def plot_distribution(self,title=None,**kwargs):
        fig,axs = plt.subplots(nrows=4,ncols=2,figsize=(12,20))
        plt.suptitle(title,fontsize=16)
        plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0.2,hspace=0.2)
        show = kwargs.pop('show',False)
        xscale = kwargs.pop('xscale',False)
        for i in range(len(self.samplers)):
            self.samplers[i].plot_distribution(
                axs = axs[i],
                show = False,
                xscale = xscale if i != 2 else 'linear',
                **kwargs,
            )
        if show:
            plt.show()
        return fig

    def sample(self, P: NDArray):
        N = P.shape[0]
        return [
            sampler.sample(N)
            for sampler in self.samplers
        ]

class BinnedMultiVar:
    def __init__(self):
        self.samplers = {}

    def add_sampler(self,P_min,P_max,sampler):
        self.samplers[(P_min,P_max)] = sampler

    def plot_sampling(self,**kwargs):
        figs = {}
        for (P_min,P_max),multi_sampler in self.samplers.items():
            figs[f'sampling_{P_min:.2f}_{P_max:.2f}'.replace('.','p')] = multi_sampler.plot_sampling(
                title = f'{P_min:.2f} < P < {P_max:.2f}',
                **kwargs,
            )
        return figs

    def plot_distribution(self,**kwargs):
        figs = {}
        for (P_min,P_max),multi_sampler in self.samplers.items():
            figs[f'dist_{P_min:.2f}_{P_max:.2f}'.replace('.','p')] = multi_sampler.plot_distribution(
                title = f'{P_min:.2f} < P < {P_max:.2f}',
                **kwargs,
            )
        return figs

    @cached_property
    def P_edges(self):
        keys = list(self.samplers.keys())
        edges = [*keys[0]]
        for key in keys[1:]:
            assert key[0] in edges, f'Missing first bin edge {key[0]} in {edges}'
            edges.append(key[1])
        return edges

    def sample(self, P: NDArray):
        keys = list(self.samplers.keys())

        #return self.samplers[keys[-1]].sample(P)

        edges = self.P_edges
        idx_P = np.maximum(
            0,
            np.minimum(
                len(edges) - 1,
                np.digitize(P,edges) - 1,
            ),
        )

        dP     = np.zeros_like(P)
        dtheta = np.zeros_like(P)
        dphi   = np.zeros_like(P)
        dr     = np.zeros_like(P)

        for i in range(len(edges) - 1):
            idx = np.where(idx_P == i)[0]
            if len(idx) == 0:
                continue
            dP_i, dtheta_i, dphi_i, dr_i = self.samplers[keys[i]].sample(P[idx])
            dP[idx] = dP_i
            dtheta[idx] = dtheta_i
            dphi[idx] = dphi_i
            dr[idx] = dr_i

        return dP,dtheta,dphi,dr




class KDESampler:
    def __init__(self,data,bandwidth,points,label):
        print (f'Performing KDE evaluation for {label}')
        start = time.time()
        self.N = data.shape[0]
        self.label = label
        self.x_grid = np.linspace(min(data), max(data), points)
        self.histpdf,_ = np.histogram(data,bins=self.x_grid,density=True)
        self.kdepdf = self.kde(data,self.x_grid,bandwidth)
        self.histcdf = self.get_cdf(self.histpdf)
        self.kdecdf = self.get_cdf(self.kdepdf)
        end = time.time()
        print (f'... done in {end-start:.2f}s')

    @staticmethod
    def kde(x, x_grid, bandwidth, **kwargs):
        """Kernel Density Estimation with Scipy"""
        #if isinstance(bandwidth,(int,float)):
        #    bandwidth /= x.std(ddof=1)
        kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
        return kde.evaluate(x_grid)

    @staticmethod
    def get_cdf(pdf):
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        return cdf

    def sample_hist(self,N):
        values = np.random.rand(N)
        value_bins = np.searchsorted(self.histcdf, values)
        random_from_cdf = self.x_grid[value_bins]
        return random_from_cdf

    def sample_kde(self,N):
        values = np.random.rand(N)
        value_bins = np.searchsorted(self.kdecdf, values)
        random_from_cdf = self.x_grid[value_bins]
        return random_from_cdf

    def plot(self):
        fig,axs = plt.subplots(1,3,figsize=(15,5))
        plt.suptitle(self.label)
        bin_centers = (self.x_grid[:-1]+self.x_grid[1:])/2

        axs[0].step(bin_centers,self.histpdf,where='mid',color='blue',label='Target')
        axs[0].step(self.x_grid,self.kdepdf,where='mid',color='orange',label='KDE')
        axs[0].set_yscale('log')
        axs[0].set_xlim(self.x_grid.min(),self.x_grid.max())
        axs[0].set_ylim(
            bottom = max(
                1e-5,
                min(
                    self.histpdf.min(),
                    self.kdepdf.min(),
                ),
            ),
            top = max(
                self.histpdf.max(),
                self.kdepdf.max(),
            )*10,
        )
        axs[0].legend()

        axs[1].step(bin_centers,self.histcdf,color='blue',label='Target')
        axs[1].step(self.x_grid,self.kdecdf,color='orange',label='KDE')
        axs[1].set_yscale('log')
        axs[1].set_xlim(self.x_grid.min(),self.x_grid.max())
        axs[1].set_ylim(
            bottom = min(
                self.histcdf.min(),
                self.kdecdf.min(),
            ),
            top = 1,
        )
        axs[1].legend()

        random_from_hist,_ = np.histogram(
            self.sample_hist(self.N),
            bins = self.x_grid,
            density = True,
        )
        random_from_kde,_ = np.histogram(
            self.sample_kde(self.N),
            bins = self.x_grid,
            density = True,
        )

        axs[2].step(bin_centers,self.histpdf,color='blue',label='Target')
        axs[2].step(bin_centers,random_from_hist,color='green',label='Sampled from hist')
        axs[2].step(bin_centers,random_from_kde,color='orange',label='Sampled from KDE')
        axs[2].set_yscale('log')
        axs[2].set_xlim(self.x_grid.min(),self.x_grid.max())
        axs[2].set_ylim(
            bottom = max(
                1e-5,
                min(
                    self.histpdf.min(),
                    random_from_hist.min(),
                    random_from_kde.min(),
                ),
            ),
            top = max(
                self.histpdf.max(),
                random_from_hist.max(),
                random_from_kde.max(),
            )*10,
        )
        axs[2].legend()

        plt.show()


