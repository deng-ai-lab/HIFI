from typing import Callable
import torch
import torch.nn as nn
import math
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj

class SLTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None, # type: ignore
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x


class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None, # type: ignore
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


class HIFINeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 0.5, v_reset: float = None, c: float = 1., gamma: float = 0., dim: int = 1, channel: int = None, surrogate_function: Callable = None, detach_reset: bool = False, cupy_fp32_inference=False, **kwargs): # type: ignore

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

        init_tau = - math.log(tau - 1.)
        self.attri_tau = torch.nn.Parameter(torch.tensor(init_tau).expand(dim, 1)) if channel is None else \
            torch.nn.Parameter(torch.tensor(init_tau).expand(channel, 1, 1))

        init_vth = - math.log(v_threshold - 1.)
        self.attri_vth = torch.nn.Parameter(torch.tensor(init_vth).expand(dim, 1)) if channel is None else \
            torch.nn.Parameter(torch.tensor(init_vth).expand(channel, 1, 1))

        if self.v_reset is not None:
            init_vr = 0. - v_reset
            self.attri_vr = torch.nn.Parameter(torch.tensor(init_vr).expand(dim, 1)) if channel is None else \
                torch.nn.Parameter(torch.tensor(init_vr).expand(channel, 1, 1))

        init_c = c
        self.attri_c = torch.nn.Parameter(torch.tensor(init_c).expand(dim, 1)) if channel is None else \
            torch.nn.Parameter(torch.tensor(init_c).expand(channel, 1, 1))

        init_gamma = gamma
        self.attri_gamma = torch.nn.Parameter(torch.tensor(init_gamma).expand(dim, 1)) if channel is None else \
            torch.nn.Parameter(torch.tensor(init_gamma).expand(channel, 1, 1))

        self.register_memory('s', 0.)

        self._g = nn.LeakyReLU(0.1)

        self._h = nn.Hardtanh(-1., 1.)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.attri_tau.sigmoid()
            vth = 1. / self.attri_vth.sigmoid()
            c = self.h(self.attri_c)
            gamma = self.attri_gamma.relu()
            if self.v_reset is not None:
                vr = self.attri_vr.tanh()
            else:
                vr = None
        # return super().extra_repr() + f', tau={tau.item()}'
        return super().extra_repr() + f', tau={tau.item()}, vth={vth.item()}, c={c.item()}, gamma={gamma.item()}, vr={vr.item() if vr is not None else None}'
    
    def neuronal_fire(self):
        return self.surrogate_function(self.v - 1 / self.attri_vth.sigmoid())
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * (1 / self.attri_vth.sigmoid())
        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * (self.v_reset + self.attri_vr.tanh())

    def neuronal_charge(self, x: torch.Tensor):
        tau = self.attri_tau.sigmoid()

        if self.decay_input:
            x = x * tau

        x -= self.attri_gamma.relu() * self.s
        x *= self._h(self.attri_c)
        

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - tau) + x
        else:
            v_reset = self.v_reset + self.attri_vr.tanh()
            if type(self.v) is float:
                self.v = v_reset * (1 - tau) + v_reset * tau + x
            else:
                self.v = self.v.detach() * (1 - tau) + v_reset * tau + x
        
        self.v = self._g(self.v)

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.s = spike.detach()
        self.neuronal_reset(spike)
        return spike
