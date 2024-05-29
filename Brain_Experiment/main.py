import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm, trange
from scipy.ndimage import gaussian_filter1d
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from spikingjelly.clock_driven.neuron import BaseNode, LIFNode

from utils import *


CLASSES =   [0, 1, 2, 3, 4, 5, 6, 7, 8]
PROB =      [1, 1, 2, 1, 2, 2, 1, 2, 1]
PROB =      [p / sum(PROB) for p in PROB]

SPIKING_RATE = [0.46579072, 0.52255960, 0.48643696, 0.52418340, 
                0.61532310, 0.41711056, 0.47511405, 0.49966460, 
                0.42397030, 0.38741988, 0.37711605, 0.39672980, 
                0.38604800, 0.34954730, 0.38668860, 0.38726908]


class ReachDataset(Dataset):
    """A simple dataset object.
    """
    def __init__(self, spikes, position, velocity, condition, **kwargs):
        self.spikes = spikes
        self.position = position
        self.velocity = velocity
        self.condition = condition
        assert len(self.spikes) == len(self.position)
        assert len(self.spikes) == len(self.velocity)
        assert len(self.spikes) == len(self.condition)

    def __len__(self):
        return self.spikes.shape[0]

    def __getitem__(self, idx):
        # Binarize the stimulus, move it and the spikes to the GPU,
        # and package into a dictionary
        return self.spikes[idx], self.position[idx], self.velocity[idx], self.condition[idx]


class GRU(nn.Module):
    def __init__(self, input_dim: int = 96, hidden_dim: int = 512, output_dim: int = 2, num_layers: int = 2):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        norm_layer = nn.LayerNorm

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            norm_layer(hidden_dim)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            norm_layer(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x):
        bs, ts, _ = x.size()
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x)

        emb = self.enc(x.view(-1, x.size(-1))).view(bs, ts, -1)

        output = []
        for t in range(ts):
            _out, hidden = self.gru(emb[:, t, :].unsqueeze(1), hidden)
            out = self.fc(_out)
            output.append(out.unsqueeze(1))
        output = torch.cat(output, dim=1)

        cond = self.cls(hidden.permute(1, 0, 2))

        return output, cond


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay, patience, num_samples):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, verbose=True, path=f'./wts/decoder/NoAug_{num_samples}.pth')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        for spikes, position, velocity, condition in train_loader:
            inputs = to_t(spikes)
            targets = to_t(velocity)
            position = to_t(position)
            condition = to_t(condition)
            optimizer.zero_grad()

            vel_out, cond_out = model(inputs)
            loss = F.smooth_l1_loss(vel_out, targets) + (1 - F.cosine_similarity(vel_out, targets, dim=2).mean())
            # add position loss according to the accumulated velocity
            start_pos = position[:, 0].unsqueeze(1)
            loss += F.smooth_l1_loss(start_pos + torch.cumsum(vel_out, axis=1), position)
            loss += F.cross_entropy(cond_out, condition.long())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for spikes, position, velocity, condition in test_loader:
                inputs = to_t(spikes)
                targets = to_t(velocity)
                position = to_t(position)
                condition = to_t(condition)

                vel_out, cond_out = model(inputs)
                start_pos = position[:, 0].unsqueeze(1)
                loss = F.smooth_l1_loss(start_pos + torch.cumsum(vel_out, axis=1), position)

                test_loss += loss.item()
        test_loss /= len(test_loader)

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


@torch.no_grad()
def plot_decoder(dataset, decoder, appendix=""):
    decoder.eval()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    r2_scores = []
    for trial in tqdm(dataset, ncols=75):
        spikes, position, velocity, condition = trial

        # decode velocity and integrate to get position
        dec_velocity = decoder(to_t(spikes.unsqueeze(0)))[0].squeeze(0)
        dec_position = position[0] + torch.cumsum((dec_velocity), axis=0).cpu()

        # gaussian filter to smooth the decoded position
        dec_position = gaussian_filter1d(from_t(dec_position), 1, axis=0)

        # calculate R² score
        r2_scores.append(r2_score(dec_position, position))

        color = sci_palette[condition]
        plt.plot()
        axs[0].plot(position[:, 0], position[:, 1], color=color, alpha=0.9)
        axs[1].plot(dec_position[:, 0], dec_position[:, 1], color=color, alpha=0.9)

    R2_score = np.mean(r2_scores)
    print(f"Average R² score: {R2_score:.4f}")
    
    axs[0].set_title("test movements")
    axs[0].set_xlabel("cursor x position")
    axs[0].set_ylabel("cursor y position")
    axs[1].set_xlabel("cursor x position")
    axs[1].set_title(f"decoded movements: R²={R2_score:.4f}")
    
    axs[0].set_xlim(-150, 150)
    axs[0].set_ylim(-150, 150)

    plt.tight_layout()

    plt.savefig(f"./fig/decoder_{appendix}.pdf")


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # input = u - Vth, if input > 0, output 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        fu = torch.tanh(input)
        fu = 1 - torch.mul(fu, fu)

        return grad_input * fu
spikeplus = STEFunction.apply


class HIFINode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.0, v_reset: float = None, c: float = 1., gamma: float = 0., dim: int = 1, surrogate_function = spikeplus, detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

        self.attri_tau = torch.nn.Parameter(torch.tensor(tau).expand(dim).clone())

        self.attri_vth = torch.nn.Parameter(torch.tensor(v_threshold).expand(dim).clone())

        if self.v_reset is not None:
            self.attri_vr = torch.nn.Parameter(torch.tensor(v_reset).expand(dim).clone())

        self.attri_c = torch.nn.Parameter(torch.tensor(c).expand(dim).clone())

        self.attri_gamma = torch.nn.Parameter(torch.tensor(gamma).expand(dim).clone())

        self.register_memory('s', 0.)

        self._g = nn.LeakyReLU(0.1)
    
    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.attri_vth)
    
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.attri_vth
        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.attri_vr

    def neuronal_charge(self, x: torch.Tensor):
        tau = 1 / self.attri_tau

        if self.decay_input:
            x = x * tau

        x += self.attri_gamma * self.s
        x *= self.attri_c

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


class SNN(nn.Module):
    """
    Build a simple fc SNN according to spikingjelly.
    """
    def __init__(self, num_classes: int = 16, time_steps: int = 33, freq: int = 15, dim: int = 96, neuron: Callable = LIFNode, pos: torch.Tensor = None, vel: torch.Tensor = None):
        super(SNN, self).__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.freq = freq
        self.dim = dim
        self.neuron_name = neuron.__name__[:-4]

        def neuron_init(neuron, dim):
            return neuron(tau=2.0, v_threshold=1.0, dim=dim) if issubclass(neuron, HIFINode) else neuron()
        
        self.linear1 = nn.Sequential(
            nn.Linear(dim, 512),
            neuron_init(neuron, 512),
            nn.LayerNorm(512),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(512, 1024),
            neuron_init(neuron, 1024),
            nn.LayerNorm(1024),
        )

        self.linear3 = nn.Sequential(
            nn.Linear(1024, 512),
            neuron_init(neuron, 512),
            nn.LayerNorm(512),
        )

        self.linear4 = nn.Sequential(
            nn.Linear(512, dim),
            neuron_init(neuron, dim),
        )

        self._generate_noise()

        self._register_output(pos, vel)

    def _generate_noise(self):
        try:
            noise = torch.load('./data/noise.pt')
            assert noise.shape == (self.num_classes, self.time_steps * self.freq, self.dim)
            print("Noise loaded.")
        except:
            noise = [(torch.rand(self.time_steps * self.freq, self.dim) < SPIKING_RATE[i]).float() for i in range(self.num_classes)]
            noise = torch.stack(noise, dim=0)
            torch.save(noise, './data/noise.pt')
            print("Noise created and saved.")

        self.register_buffer('noise', noise)

    def _register_output(self, pos_inp: torch.Tensor, vel_inp: torch.Tensor):
        try:
            pos = torch.load('./data/pos.pt')
            vel = torch.load('./data/vel.pt')
            assert pos.shape == (self.num_classes, self.time_steps, 2)
            assert vel.shape == (self.num_classes, self.time_steps, 2)
            print("Position and velocity loaded.")
        except:
            assert pos_inp is not None and vel_inp is not None
            torch.save(pos_inp, './data/pos.pt')
            torch.save(vel_inp, './data/vel.pt')
            pos = pos_inp
            vel = vel_inp
            print("Position and velocity saved.")

        self.register_buffer('pos', pos)
        self.register_buffer('vel', vel)

    def generate_spikes(self, x: torch.Tensor):
        outputs = []
        for i in range(self.time_steps * self.freq):
            x_ = x[:, i, :]
            x_ = self.linear1(x_)
            x_ = self.linear2(x_)
            x_ = self.linear3(x_)
            x_ = self.linear4(x_)
            outputs.append(x_)
        outputs = torch.stack(outputs, dim=1)

        return outputs
    
    def time_windowed(self, spikes: torch.Tensor):
        batch_size = spikes.shape[0]
        return torch.sum(spikes.view(batch_size, self.time_steps, self.freq, self.dim), dim=2)

    def forward(self): 
        return self.time_windowed(self.generate_spikes(self.noise))
    
    @torch.no_grad()
    def generate_data(self, class_idx: list, prob: float = 0.1):
        self.reset()

        conditions = torch.tensor(class_idx, device=self.noise.device)

        class_idx = [(idx - 1 if idx > 4 else torch.randint(8, 16, (1,)).item() if idx == 4 else idx) for idx in class_idx]

        noise = self.noise[class_idx]

        noise += torch.randn_like(noise) * prob

        # flip_mask = torch.bernoulli(torch.ones_like(noise) * prob).bool()
        # noise = torch.where(flip_mask, 1 - noise, noise)

        trials = self.time_windowed(self.generate_spikes(noise))

        pos = self.pos[class_idx]
        vel = self.vel[class_idx]

        return (trials, pos, vel, conditions)

    def reset(self):
        for module in self.modules():
            if isinstance(module, BaseNode):
                module.reset()

    @torch.no_grad()
    def plot_spike_raster(self):
        spikes = self.generate_spikes(self.noise)
        fig_num = spikes.shape[0]

        fig, axs = plt.subplots(fig_num, 1, figsize=(10, fig_num * 5), tight_layout=True)
        for i in range(fig_num):
            axs[i].imshow(spikes[i].T.cpu().numpy(), aspect='auto')
            axs[i].set_title(f"Class {i}")
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Neuron")
        return fig
    
    @torch.no_grad()
    def plot_trials(self):
        trials = self.forward()
        fig, axs = plt.subplots(4, 4, figsize=(20, 20), tight_layout=True)
        for i in range(4):
            for j in range(4):
                axs[i, j].imshow(trials[i * 4 + j].T.cpu().numpy(), aspect='auto')
                axs[i, j].set_title(f"Class {i * 4 + j}")
                axs[i, j].set_xlabel("Time")
                axs[i, j].set_ylabel("Neuron")
        return fig


def train_model_with_gen_data(model, generator, train_loader, test_loader, num_epochs, learning_rate, weight_decay, patience, num_samples):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, verbose=True, path=f'./wts/decoder/{generator.neuron_name}_{num_samples}.pth')

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.
        for spikes, position, velocity, condition in train_loader:
            inputs = to_t(spikes)
            targets = to_t(velocity)
            position = to_t(position)
            condition = to_t(condition)

            random_class_idx = np.random.choice(CLASSES, p=PROB, size=inputs.size(0) // 4)

            fake_inputs, fake_targets, fake_position, fake_condition = generator.generate_data(random_class_idx, 0.5)
            # concatenate the real and generated data 
            inputs = torch.cat([inputs, fake_inputs], axis=0)
            targets = torch.cat([targets, fake_targets], axis=0)
            position = torch.cat([position, fake_position], axis=0)
            condition = torch.cat([condition, fake_condition], axis=0)

            optimizer.zero_grad()

            vel_out, cond_out = model(inputs)
            loss = F.smooth_l1_loss(vel_out, targets) + (1 - F.cosine_similarity(vel_out, targets, dim=2).mean())
            # add position loss according to the accumulated velocity
            start_pos = position[:, 0].unsqueeze(1)
            loss += F.smooth_l1_loss(start_pos + torch.cumsum(vel_out, axis=1), position)
            loss += F.cross_entropy(cond_out, condition.long())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for spikes, position, velocity, condition in test_loader:
                inputs = to_t(spikes)
                targets = to_t(velocity)
                position = to_t(position)
                condition = to_t(condition)

                vel_out, cond_out = model(inputs)
                start_pos = position[:, 0].unsqueeze(1)
                loss = F.smooth_l1_loss(start_pos + torch.cumsum(vel_out, axis=1), position)

                test_loss += loss.item()
        test_loss /= len(test_loader)

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    set_seed(3407)

    ### model parameters ###
    input_dim = 96
    model_dim = 512
    output_dim = 2
    num_layers = 2

    ### training parameters ###
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 0.
    patience = 25

    ### dataset parameters ###
    num_samples = 400

    train_data, val_data, test_data = torch.load('./data/BCI_data.pt')

    for k in train_data.keys():
        train_data[k] = torch.cat([train_data[k], val_data[k]], axis=0)

    for k in train_data.keys():
        train_data[k] = train_data[k][:num_samples]

    train_dst = ReachDataset(**train_data)
    test_dst = ReachDataset(**test_data)

    train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dst, batch_size=batch_size, shuffle=False)


    ##############################################################################################################
    # training without generated data
    ##############################################################################################################


    model = GRU(input_dim=input_dim, hidden_dim=model_dim, output_dim=output_dim, num_layers=num_layers)
    model.to(device)

    train_model(model, train_loader, test_loader, num_epochs, learning_rate, weight_decay, patience, num_samples)

    model.load_state_dict(torch.load(f'./wts/decoder/NoAug_{num_samples}.pth'))
    plot_decoder(test_dst, model, f"NoAug_{num_samples}")


    ##############################################################################################################
    # training with generated data
    ##############################################################################################################

    node = HIFINode

    model = GRU(input_dim=input_dim, hidden_dim=model_dim, output_dim=output_dim, num_layers=num_layers)
    model.to(device)

    generator = SNN(neuron=node)
    print(f"Training with {generator.neuron_name}-SNN generator")
    wts = torch.load(f"./wts/generator/{generator.neuron_name}.pth", map_location='cpu')
    generator.load_state_dict(wts)
    generator.to(device)
    generator.eval()

    train_model_with_gen_data(model, generator, train_loader, test_loader, num_epochs, learning_rate, weight_decay, patience, num_samples)
    model.load_state_dict(torch.load(f'./wts/decoder/{generator.neuron_name}_{num_samples}.pth'))

    plot_decoder(test_dst, model, f"{generator.neuron_name}_{num_samples}")

