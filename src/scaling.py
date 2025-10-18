import os
import yaml
import copy
import torch
import itertools
import numpy as np
import joblib
from copy import deepcopy
from abc import ABCMeta,abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from utils import recursive_tuple_to_list

class AbsScaler(metaclass=ABCMeta):
    @abstractmethod
    def transform(self,x):
        pass

    @abstractmethod
    def inverse(self,x):
        pass

    def fit(self,x):
        pass

    def save(self,f):
        return None

    @classmethod
    def load(cls,cfg):
        return None

    @abstractmethod
    def __eq__(self,other):
        pass

class logscaler(AbsScaler):
    def transform(self,x):
        assert all(x>0)
        return torch.log(x)

    def inverse(self,x):
        return torch.exp(x)

    def save(self,f):
        return {}

    @classmethod
    def load(cls,cfg):
        return cls()

    def __eq__(self,other):
        assert type(self) is type(other), f'Other is type {type(other)}'
        return True



class logmodulus(AbsScaler):
    def transform(self,x):
        return torch.sign(x) * torch.log(1+torch.abs(x))

    def inverse(self,x):
        return torch.sign(x) * (torch.exp((x/torch.sign(x))) - 1)

    def save(self,f):
        return {}

    @classmethod
    def load(cls,cfg):
        return cls()

    def __eq__(self,other):
        assert type(self) is type(other), f'Other is type {type(other)}'
        return True


class SklearnScaler(AbsScaler):
    def __init__(self,obj):
        assert isinstance(obj,BaseEstimator), f'{obj} not from scikit-learn'
        self.obj = obj

    def fit(self,x):
        try:
            check_is_fitted(self.obj)
            raise RuntimeError('Scaler has already been fitted, this should not be called')
        except NotFittedError:
            self.obj.fit(x)

    def transform(self,x):
        # Check if fitted already #
        try:
            check_is_fitted(self.obj)
            y = self.obj.transform(x)
        except NotFittedError:
            raise RuntimeError(f'Scaler {self.obj} has not been fitted yet')
        # Sklearn produces np arrays #
        if isinstance(y,np.ndarray):
            if y.dtype == 'O': # typically when containing None
                y = y.astype(np.float32)
            y = torch.from_numpy(y)
        if x.dtype != y.dtype:
            y = y.to(x.dtype)
        return y

    def inverse(self,x):
        try:
            check_is_fitted(self.obj)
            y = self.obj.inverse_transform(x)
        except NotFittedError:
            raise RuntimeError('Scaler has not been fitted yet')
        # Sklearn produces np arrays #
        if isinstance(y,np.ndarray):
            if y.dtype == 'O': # typically when containing None
                y = y.astype(np.float32)
            y = torch.from_numpy(y)
        if x.dtype != y.dtype:
            y = y.to(x.dtype)
        return y

    def save(self,f):
        joblib.dump(self.obj,f)
        return {'obj':f}

    @classmethod
    def load(cls,cfg):
        obj = joblib.load(cfg['obj'])
        return cls(obj=obj)

    def _check_equality(self,a,b):
        if type(a) != type(b):
            return False
        if isinstance(a,dict):
            if set(a.keys()) != set(b.keys()):
                return False
            for key in a.keys():
                return self._check_equality(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                return self._check_equality(a[i],b[i])
        elif isinstance(a,np.ndarray):
            return np.allclose(a,b)
        else:
            return a == b

    def __eq__(self,other):
        assert type(self) is type(other), f'Other is type {type(other)}'
        return self._check_equality(
            self.obj.__dict__,
            other.obj.__dict__,
        )



class ScalingPipeline:
    """
        Pipeline class that applies several steps of preprocessing
    """
    def __init__(self):
        """
            Steps : ScalingStep instance list
        """
        self.steps = []

    def add_step(self,step):
        assert isinstance(step,ScalingStep)
        self.steps.append(step)

    def fit(self,names,xs,features):
        assert len(names) == len(xs)
        assert len(names) == len(features)
        for step in self.steps:
            # Find object indices that are considered in the preprocessing step #
            indices = [i for i,name in enumerate(names) if step.applies(name)]
            if len(indices) == 0:
                print (f'Skipping step {step} : applied on {step.names}, but got only {names}')
                continue
            # Fit the scaler #
            step.fit(
                names = [names[i] for i in indices],
                xs = [xs[i] for i in indices],
                features = [features[i] for i in indices],
            )
            # Apply the transform that was just fitted to get the input of the next step #
            for i in indices:
                xs[i] = step.transform(names[i],xs[i],features[i])

    def transform(self,name,x,features):
        assert x.shape[-1] == len(features), f'Mismatch between shape {x.shape} and number of features {len(features)}'
        for step in self.steps:
            if step.applies(name):
                x = step.transform(name,x,features)
        return x

    def inverse(self,name,x,features):
        assert x.shape[-1] == len(features), f'Mismatch between shape {x.shape} and number of features {len(features)}'
        for step in reversed(self.steps):
            if step.applies(name):
                x = step.inverse(name,x,features)
        return x

    def is_processed(self,feature):
        for names,step in self.steps:
            if feature in step.scaling_dict.keys():
                return True
        return False

    def __str__(self):
        return "\nPreprocessing steps\n" + "".join([str(step) for step in self.steps])

    def __eq__(self,other):
        if len(self.steps) != len(other.steps):
            return False
        for i in range(len(self.steps)):
            if self.steps[i] != other.steps[i]:
                return False
        return True

    def save(self,outdir):
        if os.path.exists(outdir):
            print (f'Will overwrite what is in output directory {outdir}')
        else:
            os.makedirs(outdir)

        configs = []
        for i,step in enumerate(self.steps):
            step_cfg = step.save(os.path.join(outdir,f'step_{i}'))
            configs.append(step_cfg)

        with open(os.path.join(outdir,'preprocessing_config.yml'),'w') as handle:
            yaml.dump(configs,handle)
        print (f'Preprocessing saved in {outdir}')

    @classmethod
    def load(cls,outdir):
        if not os.path.exists(outdir):
            raise RuntimeError(f'Could not find directory {outdir}')
        config_path = os.path.join(outdir,'preprocessing_config.yml')
        if not os.path.exists(config_path):
            raise RuntimeError(f'Could not find config {config_path}')

        with open(config_path,'r') as handle:
            config = yaml.safe_load(handle)

        inst = cls()
        for step_cfg in config:
            inst.add_step(ScalingStep.load(step_cfg,outdir))
        return inst

    @classmethod
    def equalize(cls,scalings):
        # Check number of steps #
        n_steps = [len(scaling.steps) for scaling in scalings]
        if len(set(n_steps)) != 1:
            raise RuntimeError(f'Different number of steps in scalings {n_steps}')

        # Make common scaling #
        common_scaling = cls()
        for i in range(n_steps[0]):
            common_step = ScalingStep.equalize(
                [
                    scaling.steps[i]
                    for scaling in scalings
                ]
            )
            common_scaling.add_step(common_step)
        return common_scaling


class ScalingStep:
    """
    Class that applies a scaler to some variable as determined by the scaling_dict
    """
    def __init__(self,names,scaling_dict,features_select=None): # TODO : update docs
        """
        Args :
              - scaling_dict [dict] : dict with variable name as keys, and scalers as values

            Example:
            ```
                scaling_dict = {'pt': logmodulus}
            ```

            Can also use the scikit-learn preprocessing:
            ```
                from sklearn.preprocessing import scale
                scaling_dict = {
                    'pt' : scale,
                    'eta' : scale,
                    'phi' : scale,
                    'mass' : scale,
                }
            ```
            in case of other arguments to provide, can use a lambda
            ```
                from sklearn.preprocessing import power_transform
                scaling_dict = {
                    'pt' : lambda x : power_transform(x,method='yeo-johnson'),
                    [...]
                }
            ```
        """
        # Attributes #
        if isinstance(names,str):
            self.names = [names]
        elif isinstance(names,(list,tuple)):
            self.names = names
        else:
            raise TypeError
        self.scaling_dict = scaling_dict
        self.features_select = features_select
        # Safety checks #
        if self.features_select is not None:
            if len(self.features_select) != len(self.names):
                raise RuntimeError(f'Got {len(self.names)} objects but {len(self.features_select)} set of features')
            for features in self.features_select:
                if not isinstance(features,(list,tuple)):
                    features = tuple(features)
                if len(set(features)-set(self.keys())) > 0:
                    raise RuntimeError(f'Selecting features that are not in the scaler dict {[f for f in features if f not in self.keys()]}')
        else:
            self.features_select = [tuple(self.scaling_dict.keys()) for _ in range(len(self.names))]
        for key,val in self.scaling_dict.items():
            if not isinstance(val,AbsScaler):
                raise RuntimeError(f'Scaler for key `{key}` is not AbsScaler, got `{type(val)}` instead')

    def applies(self,name):
        return name in self.names

    def keys(self):
        return self.scaling_dict.keys()

    def fit(self,names,xs,features):
        # Loop over all features of all objects #
        for feature in list(set(itertools.chain.from_iterable(features))):
            if feature in self.scaling_dict.keys():
                x = []
                for j in range(len(xs)):
                    if feature not in features[j]:
                        continue
                    if names[j] not in self.names:
                        continue
                    if feature not in self.features_select[self.names.index(names[j])]:
                        continue
                    x.append(xs[j][:,features[j].index(feature)])
                x = torch.cat(x,dim=0)
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                self.scaling_dict[feature].fit(x)

    def _process(self,name,x,features,direction):
        # Safety checks #
        assert direction in ['transform','inverse']
        assert len(features) == x.shape[1], f'Feature size is {x.shape[1]}, but received {features} feature names'
        if name not in self.names:
            return x

        x = x.clone() # avoid reference issues
        features_select = self.features_select[self.names.index(name)]

        # Loop over features #
        for i,feature in enumerate(features):
            if feature in features_select and feature in self.scaling_dict.keys():
                scaling = getattr(self.scaling_dict[feature],direction)
                x[:,i] = scaling(x[:,i].unsqueeze(-1)).squeeze(-1)
        return x

    def transform(self,name,x,features):
        """
            Process a tensor x with the preprocessing scaler (needs to be provided the features attached to each dimension)
            Args:
             - x [torch.tensor] : tensor with size [N,P,H]
             - features [list] : list of feature names (size=H)
            (N = events, P = particles, H = features)
        """
        return self._process(name,x,features,'transform')

    def inverse(self,name,x,features):
        return self._process(name,x,features,'inverse')

    def __str__(self):
        s = f"Step applied to {self.names}\n"
        max_len = max([len(feature) for feature in self.scaling_dict.keys()])
        for feature,scaler in self.scaling_dict.items():
            s += f'\t{feature:{max_len+1}s} : {scaler.__class__}\n'
        max_len = max([len(name) for name in self.names])
        for name,features in zip(self.names,self.features_select):
            s += f'  - {name:{max_len+1}s}: {features}\n'
        return s

    def __eq__(self,other):
        if set(self.names) != set(other.names):
            return False
        if self.features_select != other.features_select:
            return False
        if set(self.scaling_dict.keys()) != set(other.scaling_dict.keys()):
            return False
        for key in self.scaling_dict.keys():
            if type(self.scaling_dict[key]) is not type(other.scaling_dict[key]):
                return False
            if self.scaling_dict[key] != other.scaling_dict[key]:
                return False
        return True

    def save(self,subdir):
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        # Save the scaling_dict instances content #
        scaling_dict_cfg = {}
        for feature,instance in self.scaling_dict.items():
            instance_cfg = instance.save(os.path.join(subdir,f'{feature}.bin'))
            instance_cls = instance.__class__.__name__
            scaling_dict_cfg[feature] = {
                'class' : instance_cls,
                **instance_cfg
            }

        # Return config #
        return {
            'names': self.names,
            'scaling_dict': scaling_dict_cfg,
            'features_select' : recursive_tuple_to_list(self.features_select),
        }

    @classmethod
    def load(cls,config,subdir):
        scaling_dict = {}
        for feature,cfg in config['scaling_dict'].items():
            feature_cls = getattr(memflow.dataset.preprocessing,cfg.pop('class'))
            # need to find a cleaner way with importlib
            scaler = feature_cls.load(cfg)
            if isinstance(scaler,SklearnScaler):
                try:
                    check_is_fitted(scaler.obj)
                except NotFittedError:
                    raise RuntimeError(f'Scaler object {scaler} for feature {feature} from subdir {subdir} is not fitted, this should not happen')
            scaling_dict[feature] = scaler
        return cls(
            names = config['names'],
            scaling_dict = scaling_dict,
            features_select = config['features_select'],
        )

    @classmethod
    def equalize(cls,steps):
        # Check the names #
        # (order does not matter, so can use sets)
        names = list(set(itertools.chain.from_iterable([step.names for step in steps])))
        # Check the features_select (check if consistency if Nones)
        features_are_none = [step.features_select is None for step in steps]
        if all(features_are_none):
            # No need to check further
            features_select = None
        elif all([~is_none for is_none in features_are_none]):
            # All different than None, need to check the features
            features_select = []
            for name in names:
                features = None
                for step in steps:
                    if name in step.names:
                        step_features = step.features_select[step.names.index(name)]
                        if features is None:
                            features = step_features
                        else:
                            if features != step_features:
                                raise RuntimeError(f'At preprocessing step  of {names}, for name {name}, mismatch in field_select : {features} != {step_features}')
                features_select.append(features)
        else:
            raise RuntimeError(f'At preprocessing step of {names}, mismatch in features_select==None : {features_are_none}')
        # Check the scaling_dict #
        # First make sure we have a match in the keys/features to use #
        keys = steps[0].scaling_dict.keys()
        for j in range(1,len(steps)):
            if keys != steps[j].scaling_dict.keys():
                raise RuntimeError(f'At preprocessing step of {names}, found different set of keys between dataset 0 ({keys}) and dataset {j} ({steps[i].scaling_dict.keys()})')
        # For each key, make sure we have same class #
        for key in keys:
            classes = list(set([step.scaling_dict[key].__class__ for step in steps]))
            if len(classes) != 1:
                raise RuntimeError(f'At preprocessing step of {names}, found different scaler classes for feature {key}: {classes}')
            # if a SklearnScaler, want to makes sure it uses the same sklearn preprocessor
            if isinstance(steps[0].scaling_dict[key],SklearnScaler):
                sklearn_classes = list(set([step.scaling_dict[key].obj.__class__ for step in steps]))
                if len(sklearn_classes) != 1:
                    raise RuntimeError(f'At preprocessing step {i}, found different scikit-learn classes for feature {key}: {sklearn_classes}')

        # Reset the sklearn scalers in the scaler dict
        scaling_dict = copy.deepcopy(steps[0].scaling_dict)
        # Copy because we need to inverse the preprocessing below
        for var,scaler in scaling_dict.items():
            if isinstance(scaler,SklearnScaler):
                # Get attributes that are obtained through fit
                vars_from_fit = [
                    v for v in vars(scaler.obj) if v.endswith("_") and not v.startswith("__")
                ]
                # remove them from the attributes -> reset the fit
                for var in vars_from_fit:
                    delattr(scaler.obj,var)

        # Make combined step #
        return cls(
            names = names,
            scaling_dict = scaling_dict,
            features_select = features_select,
        )

