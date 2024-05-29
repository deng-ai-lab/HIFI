import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


sci_palette = sns.color_palette([
    "#4B6CB7",
    "#40914E",
    "#8A2BE2",
    "#D2B48C",
    "#B7B7B7",
    "#E31230",
    "#EF4026",
    "#FFD700",
    "#D07D3C"
])


line_styles = [
    '-',
    '--',
    '-.',
    ':',
    (0, (1, 10)),
    (0, (5, 10)),
    (0, (3, 1, 1, 1)),
    (0, (5, 5)),
    (0, (3, 5, 1, 5)),
]


# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.as_tensor(array, device=device, dtype=dtype) 
from_t = lambda tensor: tensor.to("cpu").detach().numpy()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def epsilon():
    return torch.tensor(1e-7)


def r2_score(predictions, actuals):
    """
        computes the r2_score
        @returns:
            computed r2_score
        
        inplementation from:
        https://github.com/mjbhobe/dl-pytorch/blob/723231b4ca57bf8a049f86f0d84f6264e2edefc4/pytorch_toolkit_bkp.py#L320
    """
    SS_res = torch.sum(torch.pow(actuals - predictions, 2))
    SS_tot = torch.sum(torch.pow(actuals - torch.mean(actuals), 2))
    return (1 - SS_res / (SS_tot + epsilon())).detach().numpy()


def compute_sta(data, channel, delay):
    spikes = data['spikes'] 
    velocity = data['velocity']
    
    num_trials, num_timesteps, num_chan = spikes.shape 

    pad_velocity = torch.cat([
        torch.zeros((num_trials, delay, 2)),
        velocity,
        torch.zeros((num_trials, delay, 2))],
        axis=1)
    
    X = torch.row_stack([
            torch.row_stack([
                torch.ravel(pad_velocity[i, t-delay:t+delay+1, :])
                for t in range(delay, delay+num_timesteps)])
            for i in range(num_trials)])
    
    Y = spikes.reshape((-1, num_chan))
    W = Y.T @ X 
    W /= Y.sum(axis=0).unsqueeze(1)
    W = W.reshape((-1, 2 * delay + 1, 2))
    return W


def plot_sta(W, delay=33):
    W_norm = torch.linalg.norm(W, axis=0)
    plt.plot(torch.arange(-delay, delay+1), W_norm[:, 0], marker='.', label='$v_x$')
    plt.plot(torch.arange(-delay, delay+1), W_norm[:, 1], marker='.', label='$v_y$')
    plt.xlim(-delay, delay)
    plt.axvline(0, ls=':', color='k')
    plt.legend()
    plt.xlabel("delay [bins]")
    plt.ylabel("spike triggered avg. velocity")
    plt.savefig('sta.pdf')
    plt.show()


def lag_spikes(spikes, lag):
    num_trials, num_timesteps, num_channels = spikes.shape
    pad = torch.zeros((num_trials, lag, num_channels))
    return torch.cat([pad, spikes], axis=1)[:, :num_timesteps, :]


'''
implement from https://github.com/gngdb/pytorch-pca/blob/main/pca.py
'''
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA_torch(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
    

'''
implement from https://github.com/google-research/google-research/blob/0eba56769505f668edf6ae2df8f4c20af5e7743c/ime/utils/tools.py#L56
'''
class EarlyStopping:
  """Class to montior the progress of the model and stop early if no improvement on validation set."""

  def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
    """Initializes parameters for EarlyStopping class.

    Args:
      patience: an integer
      verbose: a boolean
      delta: a float
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.path = path
    self.epoch = 0

  def __call__(self, val_loss, model):
    """Checks if the validation loss is better than the best validation loss.

       If so model is saved.
       If not the EarlyStopping  counter is increased
    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    self.epoch += 1

    score = -val_loss
    if self.best_score is None:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
      self.counter += 1
      if self.verbose:
        print(f"Epoch {self.epoch:03d} | EarlyStopping counter: {self.counter} out of {self.patience}")
      if self.counter >= self.patience:
        self.early_stop = True
      else:
        self.early_stop = False
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
      self.counter = 0

  def save_checkpoint(self, val_loss, model):
    """Saves the model and updates the best validation loss.

    Args:
      val_loss: a float representing validation loss
      model: the trained model
      path: a string representing the path to save the model
    """
    if self.verbose:
        print(f"Epoch {self.epoch:03d} | Valid loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
      
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss