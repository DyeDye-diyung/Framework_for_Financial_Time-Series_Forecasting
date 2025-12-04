import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=self.padding, dilation=dilation,
                              **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]


class S6(nn.Module):
    def __init__(self, d_model, state_size=64):
        super().__init__()

        self.d_model = d_model
        self.state_size = state_size

        # 1. Linear layers remain unchanged
        self.delta_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, state_size)
        self.C_proj = nn.Linear(d_model, state_size)

        # ------------------- Core Correction 1: Parameterize A to be negative -------------------
        # We initialize A as log(1), log(2), ...
        # During the forward pass, we use -exp(self.log_A) as A, ensuring it is always negative
        self.log_A = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)).repeat(d_model, 1)
        )
        # --------------------------------------------------------------------

        self.D = nn.Parameter(torch.ones(d_model))

        # ------------------- Core Correction 2: Custom initialization for delta_proj ---------
        self._init_weights()
        # --------------------------------------------------------------------

    def _init_weights(self):
        """Special initialization for the bias of delta_proj to stabilize early training"""
        # This initialization method is inspired by the official implementation to ensure a small initial delta
        nn.init.xavier_uniform_(self.delta_proj.weight)

        # Formula dt_rank = ceil(d_model / 16)
        dt_rank = math.ceil(self.d_model / 16)

        # Initialize the bias so that its output is close to log(0.001), making softplus(bias) -> delta approximately 0.001
        with torch.no_grad():
            self.delta_proj.bias.uniform_(-2, 2)
            # In Mamba, the initialization of delta is very critical. Here we simplify it,
            # just ensuring its initial value is small. For more complex initialization, refer to the official source code.
            # A simple and effective way is to make the bias close to 0, or a small negative number
            nn.init.constant_(self.delta_proj.bias, 0.0)

    def forward(self, x):
        B, L, D = x.shape

        # Use -exp(self.log_A) as A, ensuring it is negative
        A = -torch.exp(self.log_A)  # Shape: (D, N)

        delta = F.softplus(self.delta_proj(x))  # Shape: (B, L, D)
        B_ = self.B_proj(x)  # Shape: (B, L, N)
        C_ = self.C_proj(x)  # Shape: (B, L, N)

        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # Shape: (B, L, D, N)
        delta_B = torch.einsum('bld,bln->bldn', delta, B_)  # Shape: (B, L, D, N)

        h = torch.zeros(B, D, self.state_size, device=x.device)
        ys = []

        for i in range(L):
            h = delta_A[:, i] * h + rearrange(x[:, i], "b d -> b d 1") * delta_B[:, i]
            y = torch.einsum('bln,bdn->bd', C_[:, i].unsqueeze(1), h)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, state_size, conv_kernel):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.inp_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = CausalConv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel)
        self.s6 = S6(d_model=d_model, state_size=state_size)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Input x: (B, L, D)
        x_norm = self.norm(x)
        x_proj = self.inp_proj(x_norm)  # (B, L, 2*D)
        x1, x2 = x_proj.chunk(2, dim=-1)  # (B, L, D), (B, L, D)

        x1_conv = self.conv(x1.transpose(1, 2))
        x1_act = F.silu(x1_conv.transpose(1, 2))  # (B, L, D)

        x2_ssm = self.s6(x2)
        x2_act = F.silu(x2_ssm)  # (B, L, D)

        x_combined = x1_act * x2_act  # (B, L, D)

        x_out = self.out_proj(x_combined) + x  # (B, L, D)
        return x_out


class MambaTimeSeries(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int,
                 state_size: int,
                 num_layers: int,
                 conv_kernel: int,
                 prediction_horizon: int):
        """
            Initializes the MambaTimeSeries model.
            Args:
                input_dim (int): The number of features or variables in the input time series at each time step.
                output_dim (int): The number of features the model is to predict for each future time step.
                d_model (int): The internal processing dimension of the model. This is a core hyperparameter that controls the model's capacity.
                state_size (int): The size of the hidden state (h) in the S6 selective state space model.
                num_layers (int): The number of MambaBlock layers to stack. A deeper model (more layers) can capture more complex patterns.
                conv_kernel (int): The kernel size of the 1D causal convolution inside each MambaBlock.
                prediction_horizon (int): The number of future time steps to predict.
        """
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, state_size, conv_kernel) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.forecasting_head = nn.Linear(d_model, prediction_horizon * output_dim)

    def forward(self, x):
        """
        Input x: (B, L_in, F_in)
            B = batch_size
            L_in = input sequence length (lookback window)
            F_in = input feature dimension

        Output y: (B, L_out, F_out)
            L_out = prediction horizon
            F_out = output feature dimension
        """
        # (B, L_in, F_in) -> (B, L_in, D)
        x = self.input_proj(x)

        for layer in self.mamba_layers:
            x = layer(x)  # (B, L_in, D)

        x = self.norm(x)  # (B, L_in, D)
        x_last = x[:, -1, :]  # (B, D)

        # (B, D) -> (B, L_out * F_out)
        y = self.forecasting_head(x_last)

        # (B, L_out * F_out) -> (B, L_out, F_out)
        y = rearrange(y, 'b (l f) -> b l f', l=self.prediction_horizon, f=self.output_dim)
        return y
