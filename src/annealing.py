import math
import matplotlib
import matplotlib.pyplot as plt

class BaseAnnealing:
    def __init__(self,beta_min=0.,beta_max=1.,warm_up=0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.warm_up = warm_up
        assert self.warm_up >= 0

    def __call__(self,t):
        if t < self.warm_up:
            return self.beta_min
        else:
            return max(self.beta_min, self.beta_max * self.call(t-self.warm_up))

class ConstantAnnealing(BaseAnnealing):
    def call(self,t):
        return self.beta_max

class LinearAnnealing(BaseAnnealing):
    def __init__(self,T,**kwargs):
        super().__init__(**kwargs)
        self.T = T

    def call(self,t):
        return min(1.,t/self.T)

class SigmoidAnnealing(BaseAnnealing):
    def __init__(self,T,k=1.,**kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.T = T

    def call(self,t):
        return self.call(t,1/(1+math.exp(-self.k*(t-self.T))))

class CosineAnnealing(BaseAnnealing):
    def __init__(self,T,**kwargs):
        super().__init__(**kwargs)
        self.T = T

    def call(self,t):
        if t > self.T:
            return 1.0
        else:
            return 0.5*(1 - math.cos(t * math.pi / self.T))


class ExponentialAnnealing(BaseAnnealing):
    def __init__(self,k=1.,**kwargs):
        super().__init__(**kwargs)
        self.k = k

    def call(self,t):
        return 1-math.exp(-self.k * t)

class CyclicAnnealing(BaseAnnealing):
    def __init__(self,C,R,**kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.R = R
        assert isinstance(self.C,int)
        assert isinstance(self.R,int)
        assert self.R < self.C, f'R = {self.R} >= C = {self.C}'

    def call(self,t):
        return min(1,(int(t) % self.C)/self.R)


def plot_annealing(
    epochs,
    annealings,
):
    x = list(range(epochs))
    fig = plt.figure(figsize=(12,5))
    #plt.subplots_adjust(right=0.3)
    for name, annealing in annealings.items():
        plt.plot(
            x,
            [annealing(t) for t in x],
            label = name,
            linewidth = 1,
        )
    plt.xlabel('Epochs',fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
    plt.show()


