from typing import Callable
import torch
import torch.nn as nn
import math
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj

class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = None, # type: ignore
            detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


class HIFINeuron(LIFNode_sj):
    """
    HIFINeuron class represents heterogeneous spiking framework with self-inhibiting neurons.

    Args:
        tau (float): The membrane time constant of the neuron. Default is 2.
        decay_input (bool): Whether to decay the input. Default is False.
        v_threshold (float): The threshold voltage for firing. Default is 0.5.
        v_reset (float): The reset voltage after firing. Default is None.
        c (float): The scaling factor for the input. Default is 1.
        gamma (float): The scaling factor for the synaptic input. Default is 0.
        dim (int): The dimension of the neuron. Default is 1.
        learnable (bool): Whether the inner parameters are learnable. Default is True.
        surrogate_function (Callable): The surrogate function for firing. Default is None.
        detach_reset (bool): Whether to detach the reset voltage. Default is False.
        cupy_fp32_inference (bool): Whether to use cupy for FP32 inference. Default is False.
        **kwargs: Additional keyword arguments.

    Attributes:
        attri_tau (torch.Tensor): The parameter for membrane time constant.
        attri_vth (torch.Tensor): The parameter for threshold voltage.
        attri_vr (torch.Tensor): The parameter for reset voltage.
        attri_c (torch.Tensor): The parameter for input scaling factor.
        attri_gamma (torch.Tensor): The parameter for synaptic input scaling factor.

    Methods:
        extra_repr(): Returns a string representation of the neuron's parameters.
        neuronal_fire(): Calculates the firing of the neuron.
        neuronal_reset(spike): Resets the neuron after firing.
        neuronal_charge(x): Charges the neuron with input.
        forward(x): Performs the forward pass of the neuron.

    """

    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 0.5, v_reset: float = None, c: float = 1., gamma: float = 0., dim: int = 1, learnable: bool = True, surrogate_function: Callable = None, detach_reset: bool = False, cupy_fp32_inference=False, **kwargs): # type: ignore
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

        init_tau = - math.log(tau - 1.)
        self.attri_tau = self._set_param(init_tau, dim, learnable)

        init_vth = - math.log(v_threshold - 1.)
        self.attri_vth = self._set_param(init_vth, dim, learnable)

        if self.v_reset is not None:
            init_vr = v_reset
            self.attri_vr = self._set_param(init_vr, dim, learnable)

        init_c = c
        self.attri_c = self._set_param(init_c, dim, learnable)

        init_gamma = gamma
        self.attri_gamma = self._set_param(init_gamma, dim, learnable)

        self.register_memory('s', 0.)

        self._g = nn.LeakyReLU(0.1)

        self._h = nn.Hardtanh(-1., 1.)

    def _set_param(self, value, dim, learnable):
        return torch.nn.Parameter(torch.tensor(value).expand(dim, 1, 1)) if learnable else torch.tensor(value).expand(dim, 1, 1)

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
