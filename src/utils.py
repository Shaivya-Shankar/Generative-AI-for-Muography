import math
import time
import torch
from typing import Union, Optional
import numpy as np
import awkward as ak
from tqdm.notebook import tqdm
from hepunits import units
from particle import Particle

from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.utilities import move_data_to_device

MUON_MASS = Particle.from_pdgid(13).mass * units.MeV

def momentum_from_kinetic_E(kinetic, unit):
    mass = MUON_MASS / unit
    if torch.is_tensor(kinetic):
        return torch.sqrt(kinetic**2 + 2 * mass * kinetic)
    elif isinstance(kinetic,(np.ndarray,np.floating,ak.Array)):
        return np.sqrt(kinetic**2 + 2 * mass * kinetic)
    elif isinstance(kinetic, (float,int)):
        return math.sqrt(kinetic**2 + 2 * mass * kinetic)
    else:
        raise TypeError

def kinetic_E_from_momentum(momentum, unit):
    mass = MUON_MASS / unit
    if torch.is_tensor(momentum):
        return torch.sqrt(momentum**2 + mass**2) - mass
    elif isinstance(momentum,(np.ndarray,np.floating,ak.Array)):
        return np.sqrt(momentum**2 + mass**2) - mass
    elif isinstance(momentum, (float,int)):
        return math.sqrt(momentum**2 + mass**2) - mass
    else:
        raise TypeError

def concat_outputs(outputs,device='cpu'):
    # This assumes all outputs have the same type #
    types = set([type(output) for output in outputs])
    if len(types) != 1:
        raise RuntimeError(f'Outputs have different types : {types}')
    # Get type from first entry and call recursively #
    if torch.is_tensor(outputs[0]):
        return torch.cat(outputs,dim=0).to(device)
    elif isinstance(outputs[0],(list,tuple)):
        lengths = set([len(output) for output in outputs])
        if len(lengths) != 1:
            raise RuntimeError(f'Outputs are lists/tuples but have different lengths: {lengths}')
        return [
            concat_outputs([output[i] for output in outputs])
            for i in range(list(lengths)[0])
        ]
    elif isinstance(outputs[0],dict):
        keys = set([tuple(output.keys()) for output in outputs])
        if len(keys) != 1:
            raise RuntimeError(f'Outputs are dicts but have different keys: {keys}')
        return {
            key: concat_outputs([output[key] for output in outputs])
            for key in list(keys)[0]
        }
    else:
        raise NotImplementedError(f'Unknown type {type(outputs[0])}')


def predict(
    model: nn.Module,
    data: Union[
        Tensor,
        Dataset,
        DataLoader,
    ],
    batch_size: int = 1024,
    N_batch = math.inf,
    disable_tqdm: bool = False,
    device: Optional[int] = None
):
    """
    Use to produce the output of a tensor or a loader
    if batch_size is provided, will produce x sequentially

    -> avoids memory issues
    """
    model.eval()
    if device is not None:
        model = model.to(device)
    else:
        device = next(model.parameters()).device
        # if model is a torch.jit.script, cannot call model.device
    if isinstance(data,torch.Tensor) and torch.is_tensor(data):
        raise NotImplementedError('Predict does not work on tensors')
    elif isinstance(data, Dataset):
        loader = DataLoader(
            dataset = data,
            batch_size = batch_size,
            shuffle = False,
        )
        return predict(model,loader,device=device,disable_tqdm=disable_tqdm)
    elif isinstance(data, DataLoader):
        preds = []
        for idx,batch in tqdm(enumerate(data),total=min(len(data),N_batch),disable=disable_tqdm,position=0,leave=True):
            if idx >= N_batch:
                break
            batch = move_data_to_device(batch,device)
            with torch.no_grad():
                y = model(batch)
                preds.append(y)
        return concat_outputs(preds)
    else:
        raise RuntimeError(f"Unknown type {type(data)}")



@torch.jit.script
def make_rotation_matrix_3D(x: int, y: int, z: int, angle: Union[Tensor,float]) -> Tensor:
    """
        Make generic rotation matrix according to unit vector (x,y,z) and angle angle
        https://en.wikipedia.org/wiki/Transformation_matrix#Rotation_2
    """
    if isinstance(angle,float):
        angle = torch.tensor([[angle]])
    elif isinstance(angle,Tensor):
        if angle.dim() < 2:
            angle = angle.reshape(-1,1)
    else:
        raise NotImplementedError(f'Angle type is {type(angle)}, not implemented')
    assert isinstance(angle,Tensor)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    R = torch.cat(
        (
            torch.cat(
                (
                    x*x*(1-cos) + cos,
                    x*y*(1-cos) + z*sin,
                    x*z*(1-cos) - y*sin,
                ),
                dim=1,
            ).unsqueeze(-1),
            torch.cat(
                (
                    y*x*(1-cos) - z*sin,
                    y*y*(1-cos) + cos,
                    y*z*(1-cos) + x*sin,
                ),
                dim=1,
            ).unsqueeze(-1),
            torch.cat(
                (
                    z*x*(1-cos) + y*sin,
                    z*y*(1-cos) - x*sin,
                    z*z*(1-cos) + cos,
                ),
                dim=1,
            ).unsqueeze(-1),
        ),
        dim=2,
    )
    return R

class Timer:
    def __init__(self, name: str = "Timer", width: int = 20):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        self.label = f"[{self.name}]".ljust(width)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self  # so you can optionally use the context object as `t` in `with Timer() as t:`

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.label} Elapsed time: {self.elapsed:.6f} seconds")



def recursive_tuple_to_list(obj):
    if isinstance(obj,(list,tuple)):
        return [recursive_tuple_to_list(o) for o in obj]
    elif isinstance(obj,dict):
        return {key:recursive_tuple_to_list(val) for key,val in obj.items()}
    elif isinstance(obj,(float,int,str)):
        return obj
    else:
        raise TypeError

