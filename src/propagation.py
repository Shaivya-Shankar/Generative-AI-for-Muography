import torch
from typing import Tuple
from torch import Tensor
import numpy as np
import awkward as ak
import vector
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import math

vector.register_awkward()

from utils import *

class AbsPropagator:
    def __init__(self,dataset,steps):
        self.dataset = dataset
        self.X0 = dataset.X0
        self.steps = steps

    @staticmethod
    def spherical_to_cartesian(r,theta,phi):
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return x,y,z

    @staticmethod
    def cartesian_to_spherical(x,y,z):
        # Could use theta = acos(r/z) but issues with floating point
        # atan is more stable so uses theta = atan(r_xy/z) instead
        # atan2 -> to get proper sign
        r = torch.sqrt(x**2+y**2+z**2)
        theta = torch.atan2(torch.sqrt(x**2+y**2),z)
        phi = torch.atan2(y,x)
        return r,theta,phi

    def multiple_propagates(self,init_P,N,tqdm_leave=True,tqdm_position=0):
        poss,moms = [],[]
        for _ in tqdm(range(N),desc='Sampling',leave=tqdm_leave,position=tqdm_position):
            pos,mom = self.propagate(init_P,tqdm_leave=False,tqdm_position=tqdm_position+1)
            poss.append(pos)
            moms.append(mom)
        return poss,moms

    @staticmethod
    @torch.jit.script
    def update_momentum(
        P_lab_frame: Tensor,
        dP: Tensor,
        dtheta: Tensor,
        dphi: Tensor,
    ) -> Tensor:
        # Follows the same axis logic as in datasets.py : compute_variations #
        v = P_lab_frame
        # e1 = v/|v| (v = P_lab_frame) #
        e1 = torch.nn.functional.normalize(v, dim=1) # [N,3]
        # Find vector not parallel to v #
        a1 = torch.tensor([1.0, 0.0, 0.0]).expand_as(v).to(v.dtype) # [N,3]
        a2 = torch.tensor([0.0, 1.0, 0.0]).expand_as(v).to(v.dtype) # [N,3]
        # Find all the ai that are too close to parallel and switch them #
        a = torch.where(
            torch.abs(torch.einsum('ni,ni->n', e1, a1)).unsqueeze(1) > 0.999, # e1.a1
            a2,
            a1,
        ) # [N,3]
        # Compute e2 and e3 #
        e2 = a - torch.einsum('ni,ni->n', a, e1).unsqueeze(1) * e1  # [N,3]
        e2 = torch.nn.functional.normalize(e2, dim=1)
        e3 = torch.cross(e1, e2, dim=1)  # (N, 3)

        # Reconstruct new momentum (u) #
        dtheta = dtheta.unsqueeze(-1)
        dphi = dphi.unsqueeze(-1)
        u = torch.cos(dtheta) * e1 + torch.sin(dtheta) * (
            torch.cos(dphi) * e2 + torch.sin(dphi) * e3
        )

        # Apply loss of energy #
        P_dev_frame = u * (torch.norm(P_lab_frame,dim=1) - dP).unsqueeze(1)

        return P_dev_frame


    def initialise(self,init_P):
        pass

    def propagate(self,init_P,tqdm_leave=True,tqdm_position=0):
        self.initialise(init_P)
        N = init_P.shape[0]
        # Start the muons at (0,0,0)
        pos = torch.zeros((N,self.steps,3))
        # Set their momentum to (0,0,Pz=P)
        mom = torch.zeros((N,self.steps,3))
        mom[:,0,2] = init_P

        # To get from x_i to x_i+1, use the direction of P_i
        for i in tqdm(range(1,self.steps),desc='Propagation',leave=tqdm_leave,position=tqdm_position):
            # First : get new momentum
            dP, dtheta, dphi, dr = self.apply_step(step=i,P_in=mom[:,i-1,:])
            P = self.update_momentum(mom[:,i-1,:], dP, dtheta, dphi)
            # Second : get muon direction #
            n = torch.nn.functional.normalize(input=P,p=2.,dim=1)
            # Third : update position
            pos[:,i,:] = pos[:,i-1,:] + n * dr.unsqueeze(-1)
            # Last : Record momentum
            mom[:,i] = P

        # Turn into awkward arrays again #
        pos = ak.zip(
            {
                'x' : pos[:,:,0],
                'y' : pos[:,:,1],
                'z' : pos[:,:,2],
            },
            with_name = 'Vector3D',
        )
        mom = ak.zip(
            {
                'px' : mom[:,:,0],
                'py' : mom[:,:,1],
                'pz' : mom[:,:,2],
            },
            with_name = 'Momentum3D',
        )
        return pos,mom

    def plot_sampling(self,idx,show=True,N_sample=1,rng=None,quantiles=None,relative=False):
        # Make quantiles #
        if quantiles is not None:
            if N_sample <= 1:
                raise RuntimeError(f'Cannot use the quantiles with N_sample={N_sample}')
            quantiles = sorted(quantiles,reverse=True)
            tails = [round((1.-quantile)/2,5) for quantile in quantiles]
            N_q = len(tails)
            qs = np.array(tails + [0.5] + [1-t for t in tails][::-1])
            labels = [f'{quantile*100:.2f}%' for quantile in quantiles]
            colors = plt.cm.rainbow(np.linspace(0, 1, N_q))
        # Make propagatation #
        init_P = torch.tensor(self.dataset.init_P.to_numpy())[idx]
        poss,moms = self.multiple_propagates(init_P,N_sample,tqdm_leave=True)
        # Relative #
        if relative:
            poss = [
                ak.zip(
                    {
                        'x' : pos.x - self.dataset.pos3D.x[idx],
                        'y' : pos.y - self.dataset.pos3D.y[idx],
                        'z' : pos.z - self.dataset.pos3D.z[idx],
                    },
                    with_name = 'Vector3D'
                )
                for pos in poss
            ]
            moms = [
                ak.zip(
                    {
                        'px' : mom.px - self.dataset.mom3D.px[idx],
                        'py' : mom.py - self.dataset.mom3D.py[idx],
                        'pz' : mom.pz - self.dataset.mom3D.pz[idx],
                    },
                    with_name = 'Momentum3D'
                )
                for mom in moms
            ]

        # Define range #
        if rng is None:
            rng = (0,max([ak.max(ak.num(pos,axis=1)) for pos in poss]))
        x = torch.arange(rng[0],rng[1])
        figs = {}
        # Plot for each index #
        for i in range(len(idx)):
            fig,axs = plt.subplots(2,3,figsize=(16,8))
            fig.suptitle(f'Initial momentum = {self.dataset.init_P[idx[i]]:.2f} {self.dataset.energy_unit}')
            plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.3,hspace=0.3)
            for j,v in enumerate(['x','y','z']):
                if quantiles is not None:
                    # Plot quantiles #
                    qarrs = np.quantile(
                        np.concatenate(
                            [
                                np.expand_dims(
                                    getattr(poss[k],v)[i,rng[0]:rng[1]].to_numpy(),
                                    axis = 0,
                                )
                                for k in range(N_sample)
                            ]
                        ),
                        qs,
                        axis=0,
                    )
                    # Plot quantiles #
                    for k in range(N_q):
                        axs[0,j].fill_between(
                            x = x,
                            y1 = qarrs[k,rng[0]:rng[1]],
                            y2 = qarrs[len(qs)-k-1,rng[0]:rng[1]],
                            color = colors[k],
                            edgecolor = 'black',
                            linewidth = 0.5,
                            alpha = 1.,
                            label = f'Pred {labels[k]}',
                        )
                    # Plot median #
                    axs[0,j].plot(
                        x,
                        qarrs[N_q,rng[0]:rng[1]],
                        linewidth = 2,
                        linestyle = 'solid',
                        color = 'black',
                        label = 'Pred median',
                    )
                else:
                    # Plot each pred independently #
                    for k in range(N_sample):
                        axs[0,j].plot(
                            x,
                            getattr(poss[k],v)[i],
                            linewidth = 1,
                            linestyle = 'solid',
                            color = 'b',
                            label = 'Pred' if k==0 else None,
                        )
                # Plot True #
                if relative:
                    axs[0,j].plot(
                        x,
                        torch.zeros_like(x),
                        linewidth = 3,
                        linestyle = 'dashed',
                        color = 'g',
                        label = 'True',
                    )
                    axs[0,j].set_ylabel(f'${v}_{{sample}}-{v}_{{true}}$',fontsize=14)
                else:
                    axs[0,j].plot(
                        x,
                        getattr(self.dataset.pos3D,v)[idx[i],rng[0]:rng[1]],
                        linewidth = 3,
                        linestyle = 'dashed',
                        color = 'g',
                        label = 'True',
                    )
                    axs[0,j].set_ylabel(v,fontsize=14)
                axs[0,j].set_xlabel('Step',fontsize=14)
                axs[0,j].legend(loc='upper left',fontsize=8)
            for j,v in enumerate(['px','py','pz']):
                if quantiles is not None:
                    # Plot quantiles #
                    qarrs = np.quantile(
                        np.concatenate(
                            [
                                np.expand_dims(
                                    getattr(moms[k],v)[i,rng[0]:rng[1]].to_numpy(),
                                    axis = 0,
                                )
                                for k in range(N_sample)
                            ]
                        ),
                        qs,
                        axis=0,
                    )
                    # Plot quantiles #
                    for k in range(N_q):
                        axs[1,j].fill_between(
                            x = x,
                            y1 = qarrs[k,rng[0]:rng[1]],
                            y2 = qarrs[len(qs)-k-1,rng[0]:rng[1]],
                            color = colors[k],
                            edgecolor = 'black',
                            linewidth = 0.5,
                            alpha = 1.,
                            label = f'Pred {labels[k]}',
                        )
                    # Plot median #
                    axs[1,j].plot(
                        x,
                        qarrs[N_q,rng[0]:rng[1]],
                        linewidth = 2,
                        linestyle = 'solid',
                        color = 'black',
                        label = 'Pred median',
                    )
                else:
                    # Plot each pred independently #
                    for k in range(N_sample):
                        axs[1,j].plot(
                            x,
                            getattr(moms[k],v)[i,rng[0]:rng[1]],
                            linewidth = 1,
                            linestyle = 'solid',
                            color = 'b',
                            label = 'Pred' if k==0 else None,
                        )
                # Plot True #
                if relative:
                    axs[1,j].plot(
                        x,
                        torch.zeros_like(x),
                        linewidth = 3,
                        linestyle = 'dashed',
                        color = 'g',
                        label = 'True',
                    )
                    axs[1,j].set_ylabel(f'${v}_{{sample}}-{v}_{{true}}$',fontsize=14)
                else:
                    axs[1,j].plot(
                        x,
                        getattr(self.dataset.mom3D,v)[idx[i],rng[0]:rng[1]],
                        linewidth = 3,
                        linestyle = 'dashed',
                        color = 'g',
                        label = 'True',
                    )
                    axs[1,j].set_ylabel(v,fontsize=14)
                axs[1,j].set_xlabel('Step',fontsize=14)
                axs[1,j].legend(loc='upper left',fontsize=8)
            figs[f'event_{i}'] = fig
        if show:
            plt.show()
        return figs

    def plot_coverage(self,N_sample,batch_size,steps,quantiles):
        # Compute quantiles #
        if N_sample <= 1:
            raise RuntimeError(f'Cannot use the quantiles with N_sample={N_sample}')
        quantiles = sorted(quantiles)
        tails = [round((1.-quantile)/2,5) for quantile in quantiles]
        qs = np.array(tails + [1-t for t in tails][::-1])

        # Check steps #
        if isinstance(steps,int):
            steps = np.array([steps])
        elif isinstance(steps,(list,tuple)):
            steps = np.array(steps)
        elif isinstance(steps,torch.Tensor):
            steps = steps.numpy()
        elif isinstance(steps,np.ndarray):
            pass
        else:
            raise TypeError
        assert np.all(steps) >= 1
        steps -= 1 # steps start at 0

        # Do propagations per batch #
        N_tracks = int(ak.num(self.dataset.mom3D,axis=0))
        track_slices = np.r_[0:N_tracks:batch_size,N_tracks]
        counts = {
            v: [None for _ in range(len(quantiles))]
            for v in ['x','y','z','px','py','pz']
        }
        for i,f in tqdm(zip(track_slices[:-1],track_slices[1:]),total=len(track_slices)-1,desc='Batch',leave=True,position=0):
            # Propagate for batch #
            init_P = torch.tensor(self.dataset.mom3D.mag[i:f,0].to_numpy())
            poss,moms = self.multiple_propagates(init_P,N_sample,tqdm_leave=False,tqdm_position=1)
            # Compute quantiles #
            for j,v in tqdm(enumerate(['x','y','z']),total=3,desc='Position',leave=False,position=1):
                n_min_steps = ak.min(ak.count(getattr(self.dataset.pos3D,v),axis=1))
                if n_min_steps < max(steps):
                    raise RuntimeError(f'Min number of steps {n_min_steps} < requested max steps {max(steps)}')
                arr = getattr(self.dataset.pos3D,v)[i:f,steps].to_numpy()
                sample_arr = np.concatenate(
                    [
                        np.expand_dims(
                            getattr(poss[k][:,steps],v).to_numpy(),
                            axis = 0,
                        )
                        for k in range(N_sample)
                    ]
                )
                for k in tqdm(range(len(quantiles)),desc='Quantiles',total=len(quantiles),leave=False,position=2):
                    qarr = np.quantile(sample_arr,[qs[k],qs[len(qs)-k-1]],axis=0)
                    count_inside = np.logical_and(
                        arr >= qarr[0],
                        arr <= qarr[1],
                    ).sum(axis=0)
                    if counts[v][k] is None:
                        counts[v][k] = count_inside
                    else:
                        counts[v][k] += count_inside
            for j,v in tqdm(enumerate(['px','py','pz']),total=3,desc='Momentum',leave=False,position=1):
                n_min_steps = ak.min(ak.count(getattr(self.dataset.mom3D,v),axis=1))
                if n_min_steps < max(steps):
                    raise RuntimeError(f'Min number of steps {n_min_steps} < requested max steps {max(steps)}')
                arr = getattr(self.dataset.mom3D,v)[i:f,steps].to_numpy()
                sample_arr = np.concatenate(
                    [
                        np.expand_dims(
                            getattr(moms[k][:,steps],v).to_numpy(),
                            axis = 0,
                        )
                        for k in range(N_sample)
                    ]
                )
                for k in tqdm(range(len(quantiles)),desc='Quantiles',total=len(quantiles),leave=False,position=2):
                    qarr = np.quantile(sample_arr,[qs[k],qs[len(qs)-k-1]],axis=0)
                    count_inside = np.logical_and(
                        arr >= qarr[0],
                        arr <= qarr[1],
                    ).sum(axis=0)
                    if counts[v][k] is None:
                        counts[v][k] = count_inside
                    else:
                        counts[v][k] += count_inside
        # Compute efficiencies for required steps #
        pred_effs = {
            v : [
                    counts[v][k] / N_tracks
                    for k in range(len(quantiles))
            ]
            for v in ['x','y','z','px','py','pz']
        }

        # Plot coverage #
        fig,axs = plt.subplots(ncols=len(steps),figsize=(5*len(steps),4))
        if not isinstance(axs,np.ndarray):
            axs = [axs]
        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.3,hspace=0.3)

        colors = ['#bd1f01','#3f90da','#ffa90e'] * 2
        linestyles = ['solid']*3 + ['dashed']*3

        for i,step in enumerate(steps):
            for j,v in enumerate(['x','y','z','px','py','pz']):
                effs = [pred_effs[v][k][i] for k in range(len(quantiles))]
                axs[i].scatter(
                    quantiles,
                    effs,
                    color = colors[j],
                    marker = 'o',
                    s = 10,
                )
                axs[i].plot(
                    quantiles,
                    effs,
                    color = colors[j],
                    label = v,
                    linestyle = linestyles[j],
                    linewidth = 1,
                )
                axs[i].plot(
                    [0,1],
                    [0,1],
                    color = 'k',
                    linestyle = 'dashed',
                    linewidth = 2,
                )
            axs[i].set_title(f'Step {step+1}')
            axs[i].set_xlabel('True coverage')
            axs[i].set_ylabel('Predicted coverage')
            axs[i].set_xlim(0,1)
            axs[i].set_ylim(0,1)
            axs[i].legend(loc='upper left')

        plt.show()

class TruthPropagator(AbsPropagator):
    def __init__(self,sample_phi=False,**kwargs):
        # Setup phi sampling #
        self.sample_phi = sample_phi
        if self.sample_phi:
            self.dist_phi = torch.distributions.Uniform(low=-math.pi, high=math.pi)
        super().__init__(**kwargs)

    def initialise(self,init_P):
        P_all = torch.tensor(self.dataset.mom3D.mag[:,0].to_numpy())
        # Find index of init_P within P_all #
        distances = torch.cdist(init_P.float().unsqueeze(1), P_all.float().unsqueeze(1), p=2)
        self.idx = torch.min(distances, dim=1)[1].numpy()

    def apply_step(self, step:int ,P_in: Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        # Get variations #
        dP     = ak.to_torch(self.dataset.dP[self.idx,step-1])
        dtheta = ak.to_torch(self.dataset.dtheta[self.idx,step-1])
        dr     = ak.to_torch(self.dataset.dr[self.idx,step-1])
        if self.sample_phi:
            dphi = self.dist_phi.sample((P_in.shape[0],)).to(dP.dtype)
        else:
            dphi = ak.to_torch(self.dataset.dphi[self.idx,step-1])

        return dP, dtheta, dphi, dr


class HistSamplerPropagator(AbsPropagator):
    def __init__(self,sampler,**kwargs):
        self.sampler = sampler
        super().__init__(**kwargs)

    def apply_step(self,step,P_in):
        P = torch.norm(P_in,p=2.,dim=1).numpy()
        dP,dtheta,dphi,dr = self.sampler.sample(P)
        return (
            torch.tensor(dP),
            torch.tensor(dtheta),
            torch.tensor(dphi),
            torch.tensor(dr),
        )

class MDNPropagator(AbsPropagator):
    def __init__(self,model,scaling=None,**kwargs):
        self.model = model
        self.scaling = scaling
        self.dist_phi = torch.distributions.Uniform(low=-math.pi, high=math.pi)
        super().__init__(**kwargs)

    def apply_step(self,step,P_in):
        P = torch.norm(P_in,p=2.,dim=1).unsqueeze(-1)
        if self.scaling is not None:
            P = self.scaling.transform(
                name = 'input',
                x = P,
                features = ['P'],
            )
        DP = self.model.sample({'input':P},N=1).squeeze(-1)
        DP = self.scaling.inverse(
            name = 'target',
            x = DP,
            features = ['-dP','dtheta','dr']
        )
        dP,dtheta,dr = torch.split(DP,1,dim=-1)
        dphi = self.dist_phi.sample((P_in.shape[0],))
        return dP.squeeze(-1),dtheta.squeeze(-1),dphi,dr.squeeze(-1)

class VAEPropagator(AbsPropagator):
    def __init__(self,model,scaling=None,**kwargs):
        self.model = model
        self.scaling = scaling
        self.dist_phi = torch.distributions.Uniform(low=-math.pi, high=math.pi)
        super().__init__(**kwargs)

    def apply_step(self,step,P_in):
        P = torch.norm(P_in,p=2.,dim=1).unsqueeze(-1)
        X0 = torch.ones_like(P) * self.X0
        condition = torch.cat((P,X0),dim=-1)
        if self.scaling is not None:
            condition = self.scaling.transform(
                name = 'condition',
                x = condition,
                features = ['P','X0'],
            )
        DP = self.model.sample({'condition':condition},N=1).squeeze(-1).cpu()
        DP = self.scaling.inverse(
            name = 'target',
            x = DP,
            features = ['-dP','dtheta','dr']
        )
        dP,dtheta,dr = torch.split(DP,1,dim=-1)
        dphi = self.dist_phi.sample((P_in.shape[0],))
        return dP.squeeze(-1),dtheta.squeeze(-1),dphi,dr.squeeze(-1)


class NFPropagator(AbsPropagator):
    def __init__(self,model,scaling=None,**kwargs):
        self.model = model
        self.scaling = scaling
        self.dist_phi = torch.distributions.Uniform(low=-math.pi, high=math.pi)
        super().__init__(**kwargs)

    def apply_step(self,step,P_in):
        P = torch.norm(P_in,p=2.,dim=1).unsqueeze(-1)
        X0 = torch.ones_like(P) * self.X0
        condition = torch.cat((P,X0),dim=-1)
        if self.scaling is not None:
            condition = self.scaling.transform(
                name = 'condition',
                x = condition,
                features = ['P','X0'],
            )
        DP = self.model.sample({'condition':condition},N=1).squeeze(-1).cpu()
        DP = self.scaling.inverse(
            name = 'target',
            x = DP,
            features = ['-dP','dtheta','dr']
        )
        dP,dtheta,dr = torch.split(DP,1,dim=-1)
        dphi = self.dist_phi.sample((P_in.shape[0],))
        return dP.squeeze(-1),dtheta.squeeze(-1),dphi,dr.squeeze(-1)

class GANPropagator(AbsPropagator):
    """
    Propagator that uses a GAN's generator to predict each step.
    """
    def __init__(self, model, scaling=None, noise_dim=32, **kwargs):
        # The 'model' for this propagator is the GAN's generator
        super().__init__(**kwargs)
        self.model = model
        self.scaling = scaling
        self.noise_dim = noise_dim
        self.dist_phi = torch.distributions.Uniform(low=-math.pi, high=math.pi)

    def apply_step(self, step, P_in):
        # Prepare the condition tensor (Momentum P, Radiation Length X0)
        P = torch.norm(P_in, p=2., dim=1).unsqueeze(-1)
        X0 = torch.ones_like(P) * self.X0
        condition = torch.cat((P, X0), dim=-1)
        
        # Scale the condition if a scaler is provided
        if self.scaling is not None:
            condition = self.scaling.transform(
                name='condition',
                x=condition,
                features=['P', 'X0'],
            )
        
        # Generate a step using the GAN generator
        # The device should be inferred from the model's parameters
        device = next(self.model.parameters()).device
        noise = torch.randn(condition.size(0), self.noise_dim, device=device)
        DP = self.model(condition.to(device), noise).squeeze(-1).cpu()

        # Inverse transform the output to get physical values
        if self.scaling is not None:
            DP = self.scaling.inverse(
                name='target',
                x=DP,
                features=['-dP', 'dtheta', 'dr']
            )

        dP, dtheta, dr = torch.split(DP, 1, dim=-1)
        dphi = self.dist_phi.sample((P_in.shape[0],))
        
        return dP.squeeze(-1), dtheta.squeeze(-1), dphi, dr.squeeze(-1)
