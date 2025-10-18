import os
import sys
import math
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import time
import vector
import numpy as np
import uproot
import awkward as ak
import itertools
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from scipy.stats import gaussian_kde
from functools import cached_property
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

vector.register_awkward()

from hepunits import units
from particle import Particle

from constants import X0_VALUES
from distributions import HistSampler, MultiVarSampler, BinnedMultiVar
from scaling import *

MUON_MASS = Particle.from_pdgid(13).mass * units.MeV


class CombinedDataset(Dataset):
    def __init__(self,datasets,scaling=None):
        if isinstance(datasets,(tuple,list)):
            self.names = [str(i) for i in range(len(datasets))]
            self.datasets = datasets
        elif isinstance(datasets,dict):
            self.names = list(datasets.keys())
            self.datasets = list(datasets.values())
        else:
            raise RuntimeError(f'Datasets must be list or dict')

        # Compute indices #
        self.compute()

        # Get global scaling #
        if scaling is not None:
            self.scaling = scaling
        else:
            print ('Finding common scaling pipeline',end='')
            self.scaling = ScalingPipeline.equalize(
                [
                    dataset.scaling
                    for dataset in self.datasets
                ]
            )
            print ('... done',end='\n')
            # Refit tensors #
            print ('Fitting common scaling pipeline',end='')
            self.fit()
            print ('... done',end='\n')

        # Rescale all tensors #
        print ('Rescaling all tensors')
        for dataset in tqdm(self.datasets,desc='Dataset'):
            dataset.scaling = self.scaling
            dataset.scaling_to_fit = False
            dataset.scale()

    def compute(self):
        # Get attributes needed to find index per dataset #
        self.Ns = torch.tensor([len(dataset) for dataset in self.datasets])
        self.cumNs = torch.cumsum(self.Ns,dim=0)
        self.idx_dataset_start = self.cumNs-self.Ns

    def __getitem__(self, idx):
        # Find which dataset to take the index from #
        idx_of_dataset = torch.searchsorted((self.cumNs-1),idx).item()
        # Find the index within the dataset #
        idx_in_dataset = int(idx - self.idx_dataset_start[idx_of_dataset])
        # return the corresponding item #
        return {
            'material' : idx_of_dataset,
            **self.datasets[idx_of_dataset][idx_in_dataset],
        }

    def __len__(self):
        return self.cumNs[-1]

    def fit(self):
        # Safety checks #
        names = list(set([tuple(dataset.tensors.keys()) for dataset in self.datasets]))
        assert len(names) == 1, f'Different names between datasets : {names}'
        names = names[0]
        features = {
            name : list(set([tuple(dataset.features[name]) for dataset in self.datasets]))
            for name in names
        }
        for name in names:
            assert len(features[name]) == 1, f'For {name} : mismatch in features {features[name]}'
            features[name] = features[name][0]
        # Concatenate tensors #
        all_tensors = [
            torch.cat(
                [
                    dataset.objects[name][dataset.masks[name]]
                    for dataset in self.datasets
                ],
                dim = 0,
            )
            for name in names
        ]
        # Fit #
        self.scaling.fit(
            names = names,
            xs = all_tensors,
            features = [features[name] for name in names]
        )


    def compare_variations(self,show=True,bins=51):
        fig,axs = plt.subplots(ncols=5,nrows=1,figsize=(25,4))
        plt.subplots_adjust(wspace=0.3)
        logs = [True,True,False,True]
        features = ['dP','dtheta','dphi','dr']
        labels = [r'$-\Delta P$',r'$\Delta \theta$',r'$\Delta \phi$',r'$\Delta r$']
        colors = plt.cm.tab10(np.linspace(0,1,len(self.datasets)))
        for i in range(len(features)):
            values = [
                ak.ravel(getattr(dataset,features[i])).to_numpy()
                for dataset in self.datasets
            ]
            all_values = np.concatenate(values,axis=0)
            if logs[i]:
                bins_feat = np.logspace(math.log10(all_values.min()),math.log10(all_values.max()),bins)
            else:
                bins_feat = np.linspace(all_values.min(),all_values.max(),bins)
            for j in range(len(self.datasets)):
                axs[i].hist(
                    values[j],
                    bins = bins_feat,
                    color = colors[j],
                    histtype = 'step',
                )
            if logs[i]:
                axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].set_ylim(1e-1,None)
            axs[i].set_xlabel(labels[i],fontsize=14)
        names_len_max = max([len(name) for name in self.names])
        for j in range(len(self.datasets)):
            axs[-1].plot(
                [],[],
                label = f'{self.names[j]:{names_len_max+1}s} : $X_0$ = {self.datasets[j].X0:.2f}',
            )
        axs[-1].axis('off')
        axs[-1].legend(loc='center left',fontsize=16)
        if show:
            plt.show()
        return fig

    def return_dim(self,name):
        dim = None
        for i in range(len(self.datasets)):
            if name not in self.datasets[i].tensors.keys():
                raise RuntimeError(f'No `{name}` recorded in the objects of dataset {self.names[i]}')
            if dim is None:
                dim = self.datasets[i].return_dim(name)
            else:
                if dim != self.datasets[i].return_dim(name):
                    raise RuntimeError(f'Mismatch in dimension of `{name}`: {dim} != {self.datasets[i].return_dim(name)}')
        return dim

    @property
    def input_dim(self):
        return self.return_dim('input')

    @property
    def condition_dim(self):
        return self.return_dim('condition')

    @property
    def target_dim(self):
        return self.return_dim('target')

    @property
    def plotting_config(self):
        config = self.datasets[0].plotting_config
        for dataset in self.datasets[1:]:
            assert config == dataset.plotting_config, f'Mismatch in plotting_config'
        return config


class TruthDataset(Dataset):
    def __init__(
        self,
        files,
        position_vars,
        momentum_vars,
        initial_energy_var,
        position_unit,
        momentum_unit,
        initial_energy_unit,
        X0 = None,
        scaling = None,
        mask_per_track = False,
        dtype = torch.float32,
    ):
        # Super for mother class #
        super().__init__()

        # Save attributes #
        self.files = files
        self.position_vars = position_vars
        self.momentum_vars = momentum_vars
        self.position_unit = position_unit
        self.momentum_unit = momentum_unit
        self.initial_energy_var = initial_energy_var
        self.initial_energy_unit = initial_energy_unit
        self.X0 = X0
        self.mask_per_track = mask_per_track
        self.dtype = dtype

        # Public attributes #
        self.objects = {}
        self.features = {}
        self.masks = {}
        if scaling is None:
            self.scaling = ScalingPipeline()
            self.scaling_to_fit = True
        else:
            assert isinstance(scaling,ScalingPipeline)
            self.scaling = scaling
            self.scaling_to_fit = False

        # Safety checks #
        if not isinstance(self.files,(tuple,list)):
            self.files = [self.files]
        assert len(self.files) > 0, f'You have not provided any file to load'

        # Get data #
        self.extract()

        # Variations #
        self.compute_variations()

        # Apply mask to awkward arrays #
        mask = self.make_mask()
        if mask is not None:
            self.apply_mask(mask)

        # User method #
        self.process()

        # Combine tensors and masks #
        self.scale()

        # User finalize #
        self.finalize()

    @property
    def energy_unit(self):
        return 'GeV'

    @property
    def distance_unit(self):
        return 'cm'

    def process(self):
        pass

    def finalize(self):
        pass

    @property
    def energy_unit_value(self):
        return getattr(units,self.energy_unit)

    @property
    def distance_unit_value(self):
        return getattr(units,self.distance_unit)

    def make_mask(self):
        return self.dtheta > 0

    @staticmethod
    def process_file(f):
        with uproot.open(f) as F:
            treenames = list(F.keys())
            if len(treenames) > 1:
                print (f'File {f} : more than one tree, will skip {treenames[1:]}')
            arrays = {k:v.array() for k,v in F[treenames[0]].items()}
        return arrays

    @staticmethod
    def ak_to_tensor(arr,fill_value,max_no=None):
        if arr.layout.purelist_depth == 1:
            return torch.tensor(arr.to_numpy()), torch.full((ak.count(arr,axis=0),),fill_value=True)
        elif arr.layout.purelist_depth == 2:
            if max_no is None:
                max_no = ak.max(ak.num(arr, axis=1))
            arr_padded = ak.pad_none(array=arr, target=max_no, axis=1)
            mask = ~ak.is_none(arr_padded,axis=1)
            arr_filled = ak.fill_none(arr_padded, fill_value, axis=1)
            return torch.tensor(arr_filled.to_numpy()), torch.tensor(mask.to_numpy())
        else:
            raise NotImplementedError(f'Depth {arr.layout.purelist_depth} not implemented')

    @staticmethod
    def get_momemtum_coordinates(mom,coordinates='cartesian'):
        if coordinates == 'cartesian':
            return mom.px,mom.py,mom.pz
        elif coordinates == 'spherical':
            phi = mom.phi
            theta = mom.theta
            p = mom.mag
            # Replace nans in theta (usually at step 0 when track is vertical) by 0
            if ak.sum(np.isnan(theta)):
                print (f'There are {ak.sum(np.isnan(theta))} nans in theta')
                # Find positions of nan for printout #
                mask_none = ak.is_none(ak.nan_to_none(theta),axis=1)
                rows = ak.where(ak.sum(mask_none,axis=1)>0)[0]
                cols = ak.where(ak.sum(mask_none,axis=0)>0)[0]
                print (f'\trows : {rows.to_numpy()}')
                print (f'\tcols : {cols.to_numpy()}')
                # Actually replace the nans by 0 #
                theta = ak.nan_to_num(theta,nan=0.)
                assert ak.sum(np.isnan(theta)) == 0
            # Need to correct phi outside [-pi,pi] range
            shift_angle = abs(mom.phi[:,1:]-mom.phi[:,:-1]) > np.pi
            diff_py = mom.py[:,1:]-mom.py[:,:-1]
            pi_sign = ak.where(diff_py > 0,-2*np.pi,+2*np.pi)
            # No cumsum (yet) on awkward, will use loop and numpy instead
            cumsums = ak.concatenate(
                [
                    ak.unflatten(
                        np.cumsum(shift_angle[i] * pi_sign[i]),
                        counts = n,
                    )
                    for i,n in enumerate(ak.num(shift_angle,axis=1))
                ]
            )
            phi = ak.concatenate(
                (
                    phi[:,0:1],
                    #phi[:,1:] + np.cumsum(shift_angle * pi_sign, axis=1),
                    phi[:,1:] + cumsums,
                ),
                axis=1,
            )
            return p,theta,phi
        else:
            raise NotImplementedError(f'{coordinates} undefined')

    def extract(self):
        # Get array per file #
        arrays = []
        for f in tqdm(self.files,desc='Files'):
            array = self.process_file(f)
            if self.X0 is not None:
                if self.X0 == 'auto':
                    X0 = None
                    for key,val in X0_VALUES.items():
                        if key in os.path.basename(f):
                            if X0 is None:
                                X0 = ak.Array([val]*len(array[self.initial_energy_var]))
                            else:
                                raise RuntimeError(f'Clash between file {os.path.basename(f)} and X0 key values {X0_VALUES.keys()}')
                    if X0 is None:
                        raise RuntimeError(f'Culd not find between file {os.path.basename(f)} from X0 key values {X0_VALUES.keys()}')
                elif isinstance(self.X0,(float,int)):
                    X0 = ak.Array([self.X0]*len(array[self.initial_energy_var]))
                else:
                    raise NotImplementedError(f'Type {type(self.X0)} of X0 not understood')
                array['X0'] = X0
            arrays.append(array)

        # Check consistency #
        branches = set(arrays[0].keys())
        for i in range(1,len(arrays)):
            if set(arrays[i].keys()) != branches:
                raise RuntimeError(f'Mismatch between branches {branches} of file 1 and branches {set(arrays[i].keys())} of file {i+1}')
        print ('Concatenating arrays')
        self.data = {}
        for br in branches:
            self.data[br] = ak.concatenate(
                [arr[br] for arr in arrays],
                axis = 0,
            )
        print ('... done')

        # Turn into torch tensor #
        print ('Producing 3D position vector')
        self.pos3D = ak.zip(
            {
                'x': self.data[self.position_vars[0]] * self.position_unit / self.distance_unit_value,
                'y': self.data[self.position_vars[1]] * self.position_unit / self.distance_unit_value,
                'z': self.data[self.position_vars[2]] * self.position_unit / self.distance_unit_value,
            },
            with_name='Vector3D',
        )
        self.pos3D.type.show()
        print ('Producing 3D momentum vector')
        self.mom3D = ak.zip(
            {
                'px': self.data[self.momentum_vars[0]] * self.momentum_unit / self.energy_unit_value,
                'py': self.data[self.momentum_vars[1]] * self.momentum_unit / self.energy_unit_value,
                'pz': self.data[self.momentum_vars[2]] * self.momentum_unit / self.energy_unit_value,
            },
            with_name='Momentum3D',
        )
        self.mom3D.type.show()
        self.init_E = self.data[self.initial_energy_var] * self.initial_energy_unit / self.energy_unit_value
        self.init_P = self.mom3D.mag[:,0]

    def compute_variations(self):
        """ Calculate variation quantities """
        v1 = self.mom3D[:,:-1]
        v2 = self.mom3D[:,1:]
        v1.type.show()

        # Opposite of momentum loss (to have positive)
        self.dP = v1.mag - v2.mag

        # Zenith angle #
        self.dtheta = v2.deltaangle(v1)

        # Azimuthal angle #
        # Follows the same axis logic as in propagation.py : update_momentum #
        e1 = v1.unit()
        a1 = ak.zip(
            {
                'px' : ak.ones_like(v1.px),
                'py' : ak.zeros_like(v1.py),
                'pz' : ak.zeros_like(v1.pz),
            },
            with_name = 'Momentum3D',
        )
        a2 = ak.zip(
            {
                'px' : ak.zeros_like(v1.px),
                'py' : ak.ones_like(v1.py),
                'pz' : ak.zeros_like(v1.pz),
            },
            with_name = 'Momentum3D',
        )
        dot_product = e1.cross(a1)
        a = ak.where(e1.dot(a1)>0.999,a2,a1)

        e2 = a.unit() - a.unit().dot(e1) * e1
        e2 = e2.unit()
        e3 = e1.cross(e2)

        x = v2.dot(e2)
        y = v2.dot(e3)

        self.dphi = np.arctan2(y,x)

        # Step size #
        dx = self.pos3D.x[:,:-1] - self.pos3D.x[:,1:]
        dy = self.pos3D.y[:,:-1] - self.pos3D.y[:,1:]
        dz = self.pos3D.z[:,:-1] - self.pos3D.z[:,1:]
        self.dr = np.sqrt(dx**2 + dy**2 + dz**2)

    def apply_mask(self,mask):
        if self.mask_per_track:
            mask = ak.sum(mask,axis=1) == ak.count(mask,axis=1)
            print (f'Out of {ak.count(mask)} tracks, selecting {ak.sum(mask)} [{ak.sum(mask)/ak.count(mask)*100:.3f}%]')
            self.init_E = self.init_E[mask]
            self.init_P = self.init_P[mask]
            self.pos3D = ak.drop_none(ak.mask(self.pos3D,mask))
            self.mom3D = ak.drop_none(ak.mask(self.mom3D,mask))
        else:
            print (f'Out of {ak.count(mask)} steps, selecting {ak.sum(mask)} [{ak.sum(mask)/ak.count(mask)*100:.3f}%]')
            mask_ext = ak.concatenate(
                (
                    ak.Array(np.ones((ak.num(mask,axis=0),1)) > 0),
                    mask,
                ),
                axis = 1,
            )
            self.pos3D = ak.drop_none(ak.mask(self.pos3D,mask_ext))
            self.mom3D = ak.drop_none(ak.mask(self.mom3D,mask_ext))

        self.dP     = ak.drop_none(ak.mask(self.dP,mask))
        self.dtheta = ak.drop_none(ak.mask(self.dtheta,mask))
        self.dphi   = ak.drop_none(ak.mask(self.dphi,mask))
        self.dr     = ak.drop_none(ak.mask(self.dr,mask))


    def register_object(self,name,arrs,features):
        # Safety checks #
        if not isinstance(arrs,(list,tuple)):
            arrs = [arrs]
        for i,arr in enumerate(arrs):
            if not isinstance(arr,(ak.Array,vector.Vector)):
                raise RuntimeError(f'Array {i} is not awkward array/vector, is {type(arr)}')
        # Make all into tensors #
        tensor = None
        mask = None
        for arr in arrs:
            t,m = self.ak_to_tensor(arr,fill_value=0.)
            t = t.unsqueeze(-1)
            if tensor is None:
                tensor = t
                mask = m
            else:
                tensor = torch.cat([tensor,t],dim=-1)
                mask = torch.logical_and(mask,m)
        # Check features #
        assert len(features) == tensor.shape[-1],f'Provided {len(features)} features, but tensor dim is {tensor.shape}'
        # Register #
        self.objects[name] = tensor.to(self.dtype)
        self.features[name] = features
        self.masks[name] = mask

    def register_scaling(self,step):
        if self.scaling_to_fit:
            assert isinstance(step,ScalingStep)
            self.scaling.add_step(step)

    def scale(self):
        # Check number #
        if len(self.objects) == 0:
            print('No object has been recorded')
            self.tensors = None
            self.N_muons = 0
            self.N_steps = 0
            self.events = 0
        else:
            self.tensors = {}
            # Check shapes #
            shapes = set()
            for name,tensor in self.objects.items():
                shapes.add(tuple(tensor.shape[:2]))
            if len(shapes) != 1:
                raise RuntimeError(f'Tensor shape mismatch : {" ".join([f"{name}={tensor.shape}" for name,tensor in self.objects.items()])}')
            shapes = list(shapes)[0]
            self.N_muons = shapes[0]
            self.N_steps = shapes[1]
            # Fit the scaler if one step needs it #
            if self.scaling_to_fit:
                self.scaling.fit(
                    names = list(self.objects.keys()),
                    xs = [
                        self.objects[name][self.masks[name]]
                        for name in self.objects.keys()
                    ],
                    features = [self.features[name] for name in self.objects.keys()],
                )
            # Combine masks #
            common_mask = torch.tensor(
                np.logical_and.reduce(
                    list(self.masks.values())
                )
            )
            self.events = common_mask.sum()
            # Obtain the transformed tensors, and apply the common mask #
            self.tensors = {
                name : self.scaling.transform(
                    name = name,
                    x = self.objects[name][common_mask],
                    features = self.features[name],
                )
                for name in self.objects.keys()
            }

    def __len__(self):
        return self.events

    def __getitem__(self,idx):
        assert self.tensors is not None, 'No object has been recorded, cannot find item'
        return {name:tensor[idx] for name,tensor in self.tensors.items()}

    def return_dim(self,name):
        if name not in self.tensors.keys():
            raise RuntimeError(f'No `{name}` recorded in the objects')
        return self.tensors[name].shape[1]

    @property
    def input_dim(self):
        return self.return_dim('input')

    @property
    def condition_dim(self):
        return self.return_dim('condition')

    @property
    def target_dim(self):
        return self.return_dim('target')

    @staticmethod
    def _format_idx(idx):
        if not torch.is_tensor(idx):
            if not isinstance(idx,(list,tuple)):
                idx = [idx]
            idx = np.array(idx)
        else:
            if idx.dim() == 0:
                idx = idx.reshape(-1)
        return idx

    def plot_1D(self,arrs,labels,Ps=None):
        assert len(arrs) == len(labels)
        # Make figure and color palette
        ncols = len(arrs) if Ps is None else len(arrs)+1
        fig,axs = plt.subplots(
            nrows = 1,
            ncols = ncols,
            figsize = (4.5*ncols,4),
            gridspec_kw = None if Ps is None else {'width_ratios':[3]*(ncols-1)+[1]},
        )
        plt.subplots_adjust(wspace=0.3)
        if not isinstance(axs,np.ndarray):
            axs = [axs]
        Ns = [ak.num(arr,axis=0) for arr in arrs]
        if Ps is not None:
            assert max(Ns) == len(Ps)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(Ns)))
        # Plot each array #
        for i,(arr,N) in enumerate(zip(arrs,Ns)):
            for j in range(N):
                y = arr[j].to_numpy()
                x = torch.arange(len(y))
                axs[i].plot(x,y,color=colors[j])
            axs[i].set_xlabel('Steps')
            axs[i].set_ylabel(labels[i])
        # Plot legend in last ax #
        if Ps is not None:
            for i in range(max(Ns)):
                axs[-1].plot([],[],color=colors[i],label=f'P = {Ps[i]:.2f} {self.energy_unit}')
            axs[-1].axis('off')
            axs[-1].legend(loc='center')
        # Return figure #
        return fig

    def plot_2D(self,arrs,labels,Ps=None,equalize_axes=True):
        assert len(arrs) == len(labels)
        # Make figure and color palette
        idxs = list(itertools.combinations(np.arange(len(arrs)),2))
        ncols = len(idxs) if Ps is None else len(idxs)+1
        fig,axs = plt.subplots(
            nrows = 1,
            ncols = ncols,
            figsize = (3.5*ncols,4),
            gridspec_kw = None if Ps is None else {'width_ratios':[3]*(ncols-1)+[1]},
        )
        plt.subplots_adjust(wspace=0.3)
        Ns = [ak.num(arr,axis=0) for arr in arrs]
        if not isinstance(axs,np.ndarray):
            axs = [axs]
        assert all([N==max(Ns) for N in Ns]), f'Arrays must have the same number of particles, got {Ns}'
        if Ps is not None:
            assert max(Ns) == len(Ps)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(Ns)))
        # Plot each array #
        for j,(i1,i2) in enumerate(idxs):
            for i in range(max(Ns)):
                x = arrs[i1][i].to_numpy()
                y = arrs[i2][i].to_numpy()
                axs[j].plot(x,y,color=colors[i])
            axs[j].set_xlabel(labels[i1])
            axs[j].set_ylabel(labels[i2])
            if equalize_axes:
                max_xy = max(ak.max(abs(arrs[i1])),ak.max(abs(arrs[i2])))
                axs[j].set_xlim(-max_xy,max_xy)
                axs[j].set_ylim(-max_xy,max_xy)
        # Plot legend in last ax #
        if Ps is not None:
            for i in range(max(Ns)):
                axs[-1].plot([],[],color=colors[i],label=f'P = {Ps[i]:.2f} {self.energy_unit}')
            axs[-1].axis('off')
            axs[-1].legend(loc='center')
        # Return figure #
        return fig

    def plot_3D(self,arrs,labels,Ps=None):
        assert len(arrs) == 3
        assert len(arrs) == len(labels)
        # Make figure and color palette #
        if Ps is not None:
            fig = plt.figure(figsize=(8,4))
            gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[4, 1])
            ax0 = plt.subplot(gs[0], projection='3d')
            ax1 = plt.subplot(gs[1])
        else:
            fig = plt.figure(figsize=(6,4))
            ax0 = plt.axes(projection='3d')

        Ns = [ak.num(arr,axis=0) for arr in arrs]
        assert all([N==max(Ns) for N in Ns]), f'Arrays must have the same number of particles, got {Ns}'
        if Ps is not None:
            assert max(Ns) == len(Ps)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(Ns)))
        # Plot each muon #
        for i in range(max(Ns)):
            x = arrs[0][i].to_numpy()
            y = arrs[1][i].to_numpy()
            z = arrs[2][i].to_numpy()
            ax0.plot3D(
                x,y,z,
                color = colors[i],
            )
            # Plot labels in other subplot #
            if Ps is not None:
                ax1.plot(
                    [],[],
                    color = colors[i],
                    label = f'P = {Ps[i]:.2f} {self.energy_unit}',
                )
                ax1.axis('off')
        # Psthetics #
        ax0.set_xlabel(labels[0])
        ax0.set_ylabel(labels[1])
        ax0.set_zlabel(labels[2])
        if Ps is not None:
            ax1.legend(loc='center')
        max_xy = max(ak.max(abs(arrs[0])),ak.max(abs(arrs[1])))
        ax0.set_xlim(-max_xy,max_xy)
        ax0.set_ylim(-max_xy,max_xy)
        # Return figure #
        return fig

    def plot_positions(self,idx,dim='1D',show=True,legend=True):
        idx = self._format_idx(idx)
        ys = [
            getattr(self.pos3D,v)[idx]
            for v in ['x','y','z']
        ]
        if legend:
            Ps = np.take(self.init_P,indices=idx,axis=0)
        else:
            Ps = None
        labels = [
            f'Position x [{self.distance_unit}]',
            f'Position y [{self.distance_unit}]',
            f'Position z [{self.distance_unit}]',
        ]
        if dim == '1D':
            fig = self.plot_1D(ys,labels,Ps)
        elif dim == '2D':
            fig = self.plot_2D(ys[:2],labels[:2],Ps)
        elif dim == '3D':
            fig = self.plot_3D(ys,labels,Ps)
        if show:
            plt.show()
        return fig

    def plot_momentums(self,idx,dim='1D',coordinates='cartesian',show=True,legend=True):
        idx = self._format_idx(idx)
        mom = self.mom3D[idx]
        ys = self.get_momemtum_coordinates(mom,coordinates)
        if legend:
            Es = np.take(self.init_E,indices=idx,axis=0)
        else:
            Es = None
        if coordinates == 'cartesian':
            labels = [
                f'$P_x$ [{self.energy_unit}]',
                f'$P_y$ [{self.energy_unit}]',
                f'$P_z$ [{self.energy_unit}]',
            ]
        if coordinates == 'spherical':
            labels = [
                f'$P$ [{self.energy_unit}]',
                r'$\theta$ [rad]',
                r'$\phi$ [rad]',
            ]
        if dim == '1D':
            fig = self.plot_1D(ys,labels,Es)
        elif dim == '2D':
            if coordinates == 'cartesian':
                fig = self.plot_2D(ys[:2],labels[:2],Es)
            if coordinates == 'spherical':
                fig = self.plot_2D(ys[1:],labels[1:],Es,equalize_axes=False)
        elif dim == '3D':
            fig = self.plot_3D(ys,labels,Es)
        if show:
            plt.show()
        return fig

    def plot_variations(self,show=True,bins=51):
        figs = {}
        P = self.mom3D[:,:-1].mag
        bins_P = np.logspace(np.log10(ak.min(P)),np.log10(ak.max(P)),bins)
        bins_dP = np.logspace(np.log10(ak.min(self.dP)),np.log10(ak.max(self.dP)),bins)
        if ak.min(self.dtheta) == 0:
            dtheta_thresh = ak.min(self.dtheta[self.dtheta>0])
            bins_dtheta = np.r_[0,np.logspace(np.log10(dtheta_thresh),np.log10(ak.max(self.dtheta)),100)]
        else:
            dtheta_thresh = None
            bins_dtheta = np.r_[np.logspace(np.log10(ak.min(self.dtheta)),np.log10(ak.max(self.dtheta)),100)]
        bins_dphi = np.linspace(-math.pi,math.pi,bins)
        bins_dr = np.logspace(np.log10(ak.min(self.dr)),np.log10(ak.max(self.dr)),bins)

        # 1D plots #
        fig_1D,axs = plt.subplots(ncols=4,nrows=1,figsize=(18,4))
        plt.subplots_adjust(wspace=0.3)
        axs[0].hist(
            ak.ravel(self.dP).to_numpy(),
            bins = bins_dP,
            histtype = "step",
        )
        axs[0].set_xlabel(r'$-\Delta P$',fontsize=14)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')

        axs[1].hist(
            ak.ravel(self.dtheta).to_numpy(),
            bins = bins_dtheta,
            histtype = "step",
        )
        axs[1].set_xlabel(r'$\Delta \theta$',fontsize=14)
        if dtheta_thresh is None:
            axs[1].set_xscale('log')
        else:
            axs[1].set_xscale('symlog',linthresh=dtheta_thresh)
        axs[1].set_yscale('log')

        axs[2].hist(
            ak.ravel(self.dphi).to_numpy(),
            bins = bins_dphi,
            histtype = "step",
        )
        axs[2].set_xlabel(r'$\Delta \phi$',fontsize=14)

        axs[3].hist(
            ak.ravel(self.dr).to_numpy(),
            bins = bins_dr,
            histtype = "step",
        )
        axs[3].set_xlabel(r'$\Delta r$',fontsize=14)
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        figs['var_1D'] = fig_1D

        # 2D plots #
        fig_2D,axs = plt.subplots(ncols=4,nrows=1,figsize=(18,4))
        plt.subplots_adjust(wspace=0.3)
        h = axs[0].hist2d(
            ak.ravel(P).to_numpy(),
            ak.ravel(self.dP).to_numpy(),
            bins = (bins_P,bins_dP),
            norm = matplotlib.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3],ax=axs[0])
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel(r'P',fontsize=14)
        axs[0].set_ylabel(r'$\Delta P$',fontsize=14)

        h = axs[1].hist2d(
            ak.ravel(P).to_numpy(),
            ak.ravel(self.dtheta).to_numpy(),
            bins = (bins_P,bins_dtheta),
            norm = matplotlib.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3],ax=axs[1])
        axs[1].set_xscale('log')
        if dtheta_thresh is None:
            axs[1].set_yscale('log')
        else:
            axs[1].set_yscale('symlog',linthresh=dtheta_thresh)
        axs[1].set_xlabel(r'P',fontsize=14)
        axs[1].set_ylabel(r'$\Delta \theta$',fontsize=14)

        h = axs[2].hist2d(
            ak.ravel(P).to_numpy(),
            ak.ravel(self.dphi).to_numpy(),
            bins = (bins_P,bins_dphi),
            norm = matplotlib.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3],ax=axs[2])
        axs[2].set_xscale('log')
        axs[2].set_xlabel(r'P',fontsize=14)
        axs[2].set_ylabel(r'$\Delta \phi$',fontsize=14)

        h = axs[3].hist2d(
            ak.ravel(P).to_numpy(),
            ak.ravel(self.dr).to_numpy(),
            bins = (bins_P,bins_dr),
            norm = matplotlib.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3],ax=axs[3])
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        axs[3].set_xlabel(r'P',fontsize=14)
        axs[3].set_ylabel(r'$\Delta r$',fontsize=14)
        figs['var_2D'] = fig_2D

        # pairplot #
        fig_pair,axs = plt.subplots(ncols=4,nrows=4,figsize=(18,18))
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        vars = [self.dP,self.dtheta,self.dphi,self.dr]
        vars = [ak.ravel(var).to_numpy() for var in vars]
        bins = [bins_dP,bins_dtheta,bins_dphi,bins_dr]
        logs = [True,True,False,True]
        labels = [r'$-\Delta P$',r'$\Delta \theta$',r'$\Delta \phi$',r'$\Delta r$']
        for i in range(4):
            for j in range(4):
                if i < j :
                    axs[i,j].set_axis_off()
                elif i == j:
                    axs[i,j].hist(
                        vars[i],
                        bins = bins[i],
                    )
                    if logs[i]:
                        axs[i,j].set_xscale('log')
                    axs[i,j].set_yscale('log')
                    axs[i,j].set_ylim(1e-1,None)
                    axs[i,j].set_xlabel(labels[i],fontsize=14)
                else:
                    h = axs[i,j].hist2d(
                        vars[i],
                        vars[j],
                        bins = (bins[i],bins[j]),
                        norm = matplotlib.colors.LogNorm(vmin=1e-1),
                    )
                    if logs[i]:
                        axs[i,j].set_xscale('log')
                    if logs[j]:
                        axs[i,j].set_yscale('log')
                    plt.colorbar(h[3],ax=axs[i,j])
                    axs[i,j].set_xlabel(labels[i],fontsize=14)
                    axs[i,j].set_ylabel(labels[j],fontsize=14)
        figs['var_pair'] = fig_pair


        if show:
            plt.show()
        return figs


    def plot_P_loss(self,idx,dim='1D',show=True,legend=True):
        idx = self._format_idx(idx)
        y = self.mom3D[idx].mag
        y = np.cumsum(np.diff(y,axis=1),axis=1)
        if legend:
            Es = np.take(self.init_E,indices=idx,axis=0)
        else:
            Es = None
        fig = self.plot_1D([y],['$P_{step}-P_{init}$' + f' [{self.energy_unit}]'],Es)
        if show:
            plt.show()
        return fig

    def plot_variable_1D(self,name,log=False,show=True,preprocessed=True):
        if name not in self.tensors.keys():
            raise RuntimeError(f'{name} is not recorded')
        tensor = self.tensors[name]
        features = self.features[name]
        if not preprocessed:
            tensor = self.scaling.inverse(
                name = name,
                x = tensor,
                features = features,
            )
        ncols = tensor.shape[1]
        fig,axs = plt.subplots(ncols=ncols,figsize=(6*ncols,5))
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        plt.subplots_adjust(wspace=0.2)
        for i in range(ncols):
            if not preprocessed and self.plotting_config[name]['logscales'][i]:
                bins = np.logspace(math.log10(tensor[:,i].min()),math.log10(tensor[:,i].max()),51)
            else:
                bins = np.linspace(tensor[:,i].min(),tensor[:,i].max(),51)
            axs[i].hist(tensor[:,i],bins=bins,histtype='step')
            if not preprocessed:
                if self.plotting_config[name]['logscales'][i]:
                    axs[i].set_xscale('log')
            if log:
                axs[i].set_yscale('log')
                axs[i].set_ylim(1e-1,None)
            else:
                axs[i].set_ylim(0.,None)
            axs[i].set_xlabel(self.plotting_config[name]['labels'][i],fontsize=14)
        if show:
            plt.show()
        return fig

    def plot_variable_2D(self,name_x,name_y,log=False,show=True,preprocessed=True):
        if name_x not in self.tensors.keys():
            raise RuntimeError(f'{name_x} is not recorded')
        if name_y not in self.tensors.keys():
            raise RuntimeError(f'{name_y} is not recorded')
        tensor_x = self.tensors[name_x]
        tensor_y = self.tensors[name_y]
        features_x = self.features[name_x]
        features_y = self.features[name_y]
        if not preprocessed:
            tensor_x = self.scaling.inverse(
                name = name_x,
                x = tensor_x,
                features = features_x,
            )
            tensor_y = self.scaling.inverse(
                name = name_y,
                x = tensor_y,
                features = features_y,
            )

        ncols = tensor_y.shape[1]
        nrows = tensor_x.shape[1]
        fig,axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))
        if not isinstance(axs,np.ndarray):
            axs = np.array([[axs]])
        else:
            if axs.ndim == 1:
                if nrows == 1:
                    axs = axs.reshape(1,-1)
                if ncols == 1:
                    axs = axs.reshape(-1,1)
        plt.subplots_adjust(wspace=0.2,hspace=0.2)
        for i in range(nrows):
            if not preprocessed and self.plotting_config[name_x]['logscales'][i]:
                bins_x = np.logspace(math.log10(tensor_x[:,i].min()),math.log10(tensor_x[:,i].max()),51)
            else:
                bins_x = np.linspace(tensor_x[:,i].min(),tensor_x[:,i].max(),51)
            for j in range(ncols):
                if not preprocessed and self.plotting_config[name_y]['logscales'][i]:
                    bins_y = np.logspace(math.log10(tensor_y[:,j].min()),math.log10(tensor_y[:,j].max()),51)
                else:
                    bins_y = np.linspace(tensor_y[:,j].min(),tensor_y[:,j].max(),51)
                h = axs[i,j].hist2d(
                    tensor_x[:,i],
                    tensor_y[:,j],
                    bins = (bins_x,bins_y),
                    vmin = 0 if not log else None,
                    norm = matplotlib.colors.LogNorm(vmin=1e-1) if log else None,
                )
                if not preprocessed:
                    if self.plotting_config[name_x]['logscales'][i]:
                        axs[i,j].set_xscale('log')
                    if self.plotting_config[name_y]['logscales'][j]:
                        axs[i,j].set_yscale('log')
                axs[i,j].set_xlabel(self.plotting_config[name_x]['labels'][i],fontsize=14)
                axs[i,j].set_ylabel(self.plotting_config[name_y]['labels'][j],fontsize=14)
                fig.colorbar(h[3],ax=axs[i,j])
        if show:
            plt.show()
        return fig


    #def __str__(self):
    #    s = "Dataset containing\n"
    #    for key,(arr,mask) in self.data.items():
    #        s += f"\t{key:15s} : {arr.shape}\n"
    #    return s



class HistDataset(TruthDataset):
    def __init__(self,bins,make_sampler=False,interpolate_sampling=True,interpolate_cdf_factor=None,P_bins=None,**kwargs):
        # Save attributes #
        self.bins = bins
        self.make_sampler = make_sampler
        self.interpolate_sampling = interpolate_sampling
        self.interpolate_cdf_factor = interpolate_cdf_factor
        self.P_bins = P_bins

        super().__init__(**kwargs)

    def make_mask(self):
        return self.dtheta > 0

    def process(self):
        P = self.init_P
        variables = [self.dP,self.dtheta,self.dphi,self.dr]
        labels = [r'$-\Delta |P|$',r'$\Delta \theta$',r'$\Delta \phi$',r'$\Delta r$']

        if isinstance(self.bins,int):
            bins = [self.bins for _ in range(len(variables))]
        elif isinstance(self.bins,(tuple,list)):
            assert len(variables) == len(self.bins),f'{len(variables)} variables but received {len(self.bins)} bins'
            bins = self.bins
        else:
            raise TypeError(f'Bins is of type {type(self.bins)}')

        if self.make_sampler:
            if self.P_bins is not None:
                self.sampler = BinnedMultiVar()
                for P_low,P_high in zip(self.P_bins[:-1],self.P_bins[1:]):
                    # Select the diff for slice of P #
                    mask = np.logical_and(
                        P >= P_low,
                        P <  P_high,
                    )
                    if ak.sum(mask) == 0:
                        continue
                    # Make sampler #
                    self.sampler.add_sampler(
                        P_min = P_low,
                        P_max = P_high,
                        sampler = MultiVarSampler(
                            samplers = [
                                HistSampler(
                                    data = ak.ravel(
                                        ak.drop_none(
                                            ak.mask(variables[i],mask)
                                        )
                                    ).to_numpy(),
                                    bins = bins[i],
                                    label = labels[i],
                                    interpolate_sampling = self.interpolate_sampling,
                                    interpolate_cdf_factor = self.interpolate_cdf_factor,
                                )
                                for i in range(len(variables))
                            ]
                        )
                    )
            else:
                self.sampler = MultiVarSampler(
                    samplers = [
                        HistSampler(
                            data = ak.ravel(variables[i]).to_numpy(),
                            bins = bins[i],
                            label = labels[i],
                            interpolate_sampling = self.interpolate_sampling,
                            interpolate_cdf_factor = self.interpolate_cdf_factor,
                        )
                        for i in range(len(variables))
                    ]
                )

    def plot_sampling(self,**kwargs):
        if not self.make_sampler:
            raise RuntimeError('No sampler recorded, cannot plot')
        self.sampler.plot_sampling(**kwargs)


    def plot_distribution(self,**kwargs):
        if not self.make_sampler:
            raise RuntimeError('No sampler recorded, cannot plot')
        self.sampler.plot_distribution(**kwargs)


class NFDataset(TruthDataset):
    def make_mask(self):
        return self.dtheta > 0.

    @property
    def plotting_config(self):
        return {
            'condition' : {
                'features' : ['P','X0'],
                'labels' : ['$P$','$X_0$'],
                'logscales' : [True,True],
            },
            'input' : None,
            'target' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
        }

    def process(self):
        # r -> ΔP conditioned on (P,X0)

        P = self.mom3D.mag[:,:-1]
        assert self.X0 is not None and self.X0 != 0.
        X0 = ak.ones_like(P) * self.X0

        self.register_object(
            name = 'condition',
            arrs = [P,X0],
            features = ['P','X0'],
        )
        self.register_object(
            name = 'target',
            arrs = [self.dP,self.dtheta,self.dr],
            features = ['-dP','dtheta','dr'],
        )


        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': logscaler(),
                    'X0': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': SklearnScaler(StandardScaler()),
                    'X0': SklearnScaler(StandardScaler()),
                },
            )
        )

        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                    'dr'    : logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'    : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dtheta' : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dr'     : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                },
            )
        )


class MDNDataset(TruthDataset):
    def make_mask(self):
        return self.dtheta > 0.

    @property
    def plotting_config(self):
        return {
            'input' : {
                'features' : ['P','X0'],
                'labels' : ['$P$','$X_0$'],
                'logscales' : [True,True],
            },
            'condition' : None,
            'target' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
        }

    def process(self):
        P = self.mom3D.mag[:,:-1]
        assert self.X0 is not None and self.X0 != 0.
        X0 = ak.ones_like(P) * self.X0

        self.register_object(
            name = 'input',
            arrs = [P,X0],
            features = ['P','X0'],
        )
        self.register_object(
            name = 'target',
            arrs = [self.dP,self.dtheta,self.dr],
            features = ['-dP','dtheta','dr'],
        )

        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    'P': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    'P': SklearnScaler(StandardScaler()),
                },
            )
        )

        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                    'dr'    : logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'    : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dtheta' : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dr'     : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                },
            )
        )

class VAEDataset(TruthDataset):
    def make_mask(self):
        return self.dtheta > 0.

    @property
    def plotting_config(self):
        return {
            'condition' : {
                'features' : ['P','X0'],
                'labels' : ['$P$','$X_0$'],
                'logscales' : [True,True],
            },
            'input' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
            'target' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
        }

    def process(self):
        # ΔP -> mu,sigma x z -> ΔP conditioned on (P,X0)

        P = self.mom3D.mag[:,:-1]
        assert self.X0 is not None and self.X0 != 0.
        X0 = ak.ones_like(P) * self.X0

        self.register_object(
            name = 'condition',
            arrs = [self.mom3D.mag[:,:-1],X0],
            features = ['P','X0'],
        )
        self.register_object(
            name = 'input',
            arrs = [self.dP,self.dtheta,self.dr],
            features = ['-dP','dtheta','dr'],
        )
        self.register_object(
            name = 'target',
            arrs = [self.dP,self.dtheta,self.dr],
            features = ['-dP','dtheta','dr'],
        )


        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': logscaler(),
                    'X0': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': SklearnScaler(StandardScaler()),
                    'X0': SklearnScaler(StandardScaler()),
                },
            )
        )

        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                    'dr'    : logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    '-dP'    : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dtheta' : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dr'     : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                    'dr'    : logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'target',
                scaling_dict = {
                    '-dP'    : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dtheta' : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                    'dr'     : SklearnScaler(MinMaxScaler(feature_range=(-5,+5))),
                },
            )
        )


class ClassifierDataset(TruthDataset):
    def make_mask(self):
        mask = np.logical_and.reduce(
            (
                self.dtheta > 0.,
                self.dP <= 10,
            )
        )
        return mask

    @property
    def plotting_config(self):
        return {
            'input' : {
                'features' : ['-dP','dtheta'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$'],
                'logscales' : [True,True,False],
            },
            'target' : {
                'features' : ['class'],
                'labels' : [r'class'],
                'logscales' : [False],
            },
        }

    def process(self):
        self.register_object(
            name = 'input',
            arrs = [self.dP,self.dtheta],
            features = ['-dP','dtheta'],
        )
        self.register_object(
            name = 'target',
            arrs = [
                (abs((self.dr / self.X0 / 0.01) - 1) < 1e-3) * 1,
            ],
            features = ['class'],
        )



        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': SklearnScaler(StandardScaler()),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'input',
                scaling_dict = {
                    '-dP'    : SklearnScaler(StandardScaler()),
                    'dtheta' : SklearnScaler(StandardScaler()),
                },
            )
        )

class GANDataset(TruthDataset):
    def make_mask(self):
        """
        Create a mask to select valid physics steps.
        """
        return self.dtheta > 0.

    @property
    def plotting_config(self):
        """
        Configuration for plotting the dataset variables. This helps in automatic
        labeling and scaling of axes in your plotting functions.
        """
        return {
            'condition' : {
                'features' : ['P','X0'],
                'labels' : ['$P$','$X_0$'],
                'logscales' : [True,True],
            },
            'input' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
            'target' : {
                'features' : ['-dP','dtheta','dr'],
                'labels' : [r'$-\Delta P$',r'$\Delta\theta$',r'$\Delta r$'],
                'logscales' : [True,True,True],
            },
        }

    def process(self):
        """
        Process the raw awkward arrays into tensors suitable for the GAN.
        This involves identifying the conditional variables and the real data (input/target).
        """
        # --- Define Conditional and Target Variables ---
        P = self.mom3D.mag[:,:-1]
        assert self.X0 is not None and self.X0 != 0., "X0 (radiation length) must be set for the dataset"
        X0 = ak.ones_like(P) * self.X0

        # --- Register Tensors ---
        # The 'condition' for both Generator and Discriminator
        self.register_object(
            name = 'condition',
            arrs = [P, X0],
            features = ['P','X0'],
        )
        # The 'real' data samples for the Discriminator
        self.register_object(
            name = 'input',
            arrs = [self.dP, self.dtheta, self.dr],
            features = ['-dP', 'dtheta', 'dr'],
        )
        # For GANs, the target is the same as the input for real samples
        self.register_object(
            name = 'target',
            arrs = [self.dP, self.dtheta, self.dr],
            features = ['-dP', 'dtheta', 'dr'],
        )

        # --- Register Scaling Steps ---
        # Log-scale and standardize the conditional variables
        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': logscaler(),
                    'X0': logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = 'condition',
                scaling_dict = {
                    'P': SklearnScaler(StandardScaler()),
                    'X0': SklearnScaler(StandardScaler()),
                },
            )
        )

        # Log-scale and then apply MinMax scaling to the target variables
        # This bounds the outputs, which can help with training stability.
        self.register_scaling(
            ScalingStep(
                names = ['input', 'target'], # Apply to both input and target
                scaling_dict = {
                    '-dP'   : logscaler(),
                    'dtheta': logscaler(),
                    'dr'    : logscaler(),
                },
            )
        )
        self.register_scaling(
            ScalingStep(
                names = ['input', 'target'], # Apply to both input and target
                scaling_dict = {
                    '-dP'    : SklearnScaler(MinMaxScaler(feature_range=(-1, 1))),
                    'dtheta' : SklearnScaler(MinMaxScaler(feature_range=(-1, 1))),
                    'dr'     : SklearnScaler(MinMaxScaler(feature_range=(-1, 1))),
                },
            )
        )

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import awkward as ak

class GANDataset_test(TruthDataset):
    def __init__(self, *args, transformations=['MinMax'], **kwargs):
        """
        Initialize the GANDataset with transformation types that map to [-1, 1].
        
        Args:
            transformations: List of transformation types 
                           ['MinMax', 'LogMinMax', 'Tanh', 'Sigmoid', 'LinearScale', 'LogStandardMinMax']
        """
        # SET TRANSFORMATIONS FIRST before calling super().__init__()
        self.transformations = transformations
        self.transformed_data = {}
        
        # Now call parent constructor
        super().__init__(*args, **kwargs)
        
    def make_mask(self):
        """
        Create a mask to select valid physics steps.
        """
        return self.dtheta > 0.
    
    @property
    def plotting_config(self):
        """
        Configuration for plotting the dataset variables.
        """
        return {
            'condition': {
                'features': ['P', 'X0'],
                'labels': ['$P$', '$X_0$'],
                'logscales': [True, True],
            },
            'input': {
                'features': ['-dP', 'dtheta', 'dr'],
                'labels': [r'$-\Delta P$', r'$\Delta\theta$', r'$\Delta r$'],
                'logscales': [True, True, True],
            },
            'target': {
                'features': ['-dP', 'dtheta', 'dr'],
                'labels': [r'$-\Delta P$', r'$\Delta\theta$', r'$\Delta r$'],
                'logscales': [True, True, True],
            },
        }
    
    def apply_log_transform(self, data):
        """
        Apply logarithmic transformation to data, handling negative values.
        """
        # Ensure positive values for log transformation
        data = np.abs(data)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        return np.log(data + epsilon)
    
    def safe_awkward_to_numpy(self, awkward_array):
        """
        Safely convert awkward array to numpy, handling jagged arrays.
        """
        try:
            # Try direct conversion first
            return ak.to_numpy(awkward_array)
        except ValueError as e:
            if "cannot convert to RegularArray" in str(e):
                # Array is jagged, flatten it first
                flattened = ak.flatten(awkward_array)
                return ak.to_numpy(flattened)
            else:
                raise e
    
    def apply_single_transformation(self, data, transformation_type):
        """
        Apply a single transformation to the data that maps to [-1, 1].
        
        Args:
            data: Input data array
            transformation_type: Type of transformation to apply
            
        Returns:
            Transformed data array in range [-1, 1]
        """
        
        if transformation_type == 'MinMax':
            """Standard MinMax scaling to [-1, 1]"""
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            normalized = scaler.fit_transform(data)
            if normalized.shape[1] == 1:
                normalized = normalized.flatten()
            return normalized
        
        elif transformation_type == 'LogMinMax':
            """Log transformation followed by MinMax scaling to [-1, 1]"""
            log_data = self.apply_log_transform(data)
            if len(log_data.shape) == 1:
                log_data = log_data.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized = scaler.fit_transform(log_data)
            if normalized.shape[1] == 1:
                normalized = normalized.flatten()
            return normalized
        
        elif transformation_type == 'LogStandardMinMax':
            """Log + Standard + MinMax scaling to [-1, 1]"""
            # Step 1: Log transformation
            log_data = self.apply_log_transform(data)
            if len(log_data.shape) == 1:
                log_data = log_data.reshape(-1, 1)
            
            # Step 2: Standard scaling (mean=0, std=1)
            standard_scaler = StandardScaler()
            standardized = standard_scaler.fit_transform(log_data)
            
            # Step 3: MinMax scaling to [-1, 1]
            minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized = minmax_scaler.fit_transform(standardized)
            
            if normalized.shape[1] == 1:
                normalized = normalized.flatten()
            return normalized
        
        elif transformation_type == 'Tanh':
            """Tanh scaling - naturally maps to (-1, 1)"""
            # Normalize data first to prevent saturation
            data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
            return np.tanh(data_normalized)
        
        elif transformation_type == 'Sigmoid':
            """Sigmoid-based scaling to (-1, 1)"""
            # Normalize data first to prevent saturation
            data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
            sigmoid = 1 / (1 + np.exp(-data_normalized))
            return 2 * sigmoid - 1  # Map [0,1] to [-1,1]
        
        elif transformation_type == 'LinearScale':
            """Custom linear scaling to [-1, 1]"""
            data_min, data_max = np.min(data), np.max(data)
            if data_max == data_min:
                return np.zeros_like(data)  # Handle constant data
            return 2 * (data - data_min) / (data_max - data_min) - 1
        
        elif transformation_type == 'LogTanh':
            """Log transformation followed by Tanh scaling"""
            log_data = self.apply_log_transform(data)
            log_normalized = (log_data - np.mean(log_data)) / (np.std(log_data) + 1e-8)
            return np.tanh(log_normalized)
        
        elif transformation_type == 'LogSigmoid':
            """Log transformation followed by Sigmoid scaling"""
            log_data = self.apply_log_transform(data)
            log_normalized = (log_data - np.mean(log_data)) / (np.std(log_data) + 1e-8)
            sigmoid = 1 / (1 + np.exp(-log_normalized))
            return 2 * sigmoid - 1
            
        else:
            raise ValueError(f"Unsupported transformation: {transformation_type}. "
                           f"Available: {self.list_available_transformations()}")
    
    def apply_transformations_by_X0(self, data_dict, X0_values):
        """
        Apply multiple transformations to data grouped by X0 values.
        """
        # Convert to numpy arrays if needed (with safe conversion)
        if hasattr(X0_values, 'to_numpy') or str(type(X0_values)).startswith('<Array'):
            X0_np = self.safe_awkward_to_numpy(X0_values)
        else:
            X0_np = np.array(X0_values)
        
        # Get unique X0 values
        unique_X0_values = np.unique(X0_np)
        results = {}
        
        # Process each unique X0 value
        for x0_val in unique_X0_values:
            mask = X0_np == x0_val
            results[float(x0_val)] = {}
            
            # Apply each transformation to the data subset
            for trans in self.transformations:
                results[float(x0_val)][trans] = {}
                
                # Apply transformation to each feature in the data dictionary
                for feature_name, feature_data in data_dict.items():
                    # Convert to numpy safely
                    if hasattr(feature_data, 'to_numpy') or str(type(feature_data)).startswith('<Array'):
                        feature_np = self.safe_awkward_to_numpy(feature_data)
                    else:
                        feature_np = np.array(feature_data)
                    
                    feature_subset = feature_np[mask]
                    
                    # Apply the specific transformation
                    normalized = self.apply_single_transformation(feature_subset, trans)
                    results[float(x0_val)][trans][feature_name] = normalized
        
        return results
    
    def process(self):
        """
        Process the raw awkward arrays into tensors suitable for the GAN.
        Now includes grouping by X0 and applying multiple transformations.
        """
        # --- Define Conditional and Target Variables ---
        P = self.mom3D.mag[:, :-1]
        assert self.X0 is not None and self.X0 != 0., "X0 (radiation length) must be set for the dataset"
        X0 = ak.ones_like(P) * self.X0
        
        # --- Create data dictionaries for processing ---
        condition_data = {
            'P': P,
            'X0': X0
        }
        
        input_target_data = {
            '-dP': self.dP,
            'dtheta': self.dtheta,
            'dr': self.dr
        }
        
        # --- FIXED: Safe conversion of awkward arrays to numpy ---
        condition_data_np = {}
        for k, v in condition_data.items():
            if hasattr(v, 'to_numpy') or str(type(v)).startswith('<Array'):
                condition_data_np[k] = self.safe_awkward_to_numpy(v)
            else:
                condition_data_np[k] = np.array(v)
        
        input_target_data_np = {}
        for k, v in input_target_data.items():
            if hasattr(v, 'to_numpy') or str(type(v)).startswith('<Array'):
                input_target_data_np[k] = self.safe_awkward_to_numpy(v)
            else:
                input_target_data_np[k] = np.array(v)
        
        # Get X0 values for grouping (now safely converted)
        X0_values = condition_data_np['X0']
        
        # Apply multiple transformations grouped by X0
        self.transformed_data = {
            'condition': self.apply_transformations_by_X0(condition_data_np, X0_values),
            'input': self.apply_transformations_by_X0(input_target_data_np, X0_values),
            'target': self.apply_transformations_by_X0(input_target_data_np, X0_values)  # Same as input for GANs
        }
        
        # --- Register original objects (for compatibility) ---
        self.register_object(
            name='condition',
            arrs=[P, X0],
            features=['P', 'X0'],
        )
        
        self.register_object(
            name='input',
            arrs=[self.dP, self.dtheta, self.dr],
            features=['-dP', 'dtheta', 'dr'],
        )
        
        self.register_object(
            name='target',
            arrs=[self.dP, self.dtheta, self.dr],
            features=['-dP', 'dtheta', 'dr'],
        )
    
    def get_transformed_data(self, data_type=None, X0_value=None, transformation=None):
        """
        Retrieve transformed data with optional filtering.
        """
        if not hasattr(self, 'transformed_data'):
            raise ValueError("Data not processed yet. Call process() first.")
        
        result = self.transformed_data
        
        if data_type is not None:
            result = {data_type: result[data_type]}
        
        if X0_value is not None:
            filtered_result = {}
            for dt, dt_data in result.items():
                if X0_value in dt_data:
                    filtered_result[dt] = {X0_value: dt_data[X0_value]}
            result = filtered_result
        
        if transformation is not None:
            filtered_result = {}
            for dt, dt_data in result.items():
                filtered_result[dt] = {}
                for x0_val, x0_data in dt_data.items():
                    if transformation in x0_data:
                        filtered_result[dt][x0_val] = {transformation: x0_data[transformation]}
            result = filtered_result
        
        return result
    
    def list_available_transformations(self):
        """
        List all available transformation types that map to [-1, 1].
        """
        return ['MinMax', 'LogMinMax', 'LogStandardMinMax', 'Tanh', 'Sigmoid', 'LinearScale', 'LogTanh', 'LogSigmoid']