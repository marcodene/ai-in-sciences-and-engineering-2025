import torch
import torch.nn as nn

class FILM(torch.nn.Module):
    def __init__(self,
                channels,
                use_bn = True):
        super(FILM, self).__init__()
        self.channels = channels

        self.inp2scale = nn.Linear(in_features=1, out_features=channels, bias=True)
        self.inp2bias = nn.Linear(in_features=1, out_features=channels, bias=True)

        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(0)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)

        if use_bn:
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x, time):
        """
        Args:
            x: (batch, channels, spatial) - features
            time: (batch,) - scalar time per sample
        
        Returns:
            Modulated features: x * (1 + scale(t)) + bias(t)
        """
        x = self.norm(x)
        time = time.reshape(-1, 1).type_as(x)  # (batch, 1)
        scale = self.inp2scale(time)  # (batch, channels)
        bias = self.inp2bias(time)    # (batch, channels)
        scale = scale.unsqueeze(2).expand_as(x)  # (batch, channels, spatial)
        bias  = bias.unsqueeze(2).expand_as(x)   # (batch, channels, spatial)

        return x * (1. + scale) + bias


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1, 
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], 
            self.weights
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, in_dim=3, out_dim=1, padding=0.25):
        super(FNO1d, self).__init__()

        self.modes = modes
        self.width = width
        self.padding = padding  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(in_dim, self.width)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.film1 = FILM(channels=width, use_bn=True)
        self.film2 = FILM(channels=width, use_bn=True)
        self.film3 = FILM(channels=width, use_bn=True)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, out_dim)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer, film, time):
        x = spectral_layer(x) + conv_layer(x)
        x = film(x, time)
        x = self.activation(x)
        return x

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        """
        Args:
            x: (batch, spatial, 3) - Input features [x-coords, u(x), Δt]
        
        Returns:
            Output: (batch, spatial, 1) - Predicted u(x) evolved by Δt
        """
        # Extract time from input features (constant across spatial dimension)
        time = x[:, 0, 2]  # (batch,) - Δt from first spatial point
        
        x = self.linear_p(x)  # (batch, spatial, width)
        x = x.permute(0, 2, 1)  # (batch, width, spatial)

        # Apply Fourier layers with time conditioning
        x = self.fourier_layer(x, self.spect1, self.lin0, self.film1, time)
        x = self.fourier_layer(x, self.spect2, self.lin1, self.film2, time)
        x = self.fourier_layer(x, self.spect3, self.lin2, self.film3, time)
        
        x = x.permute(0, 2, 1)  # (batch, spatial, width)
        x = self.linear_layer(x, self.linear_q)  # (batch, spatial, 32)
        x = self.output_layer(x)  # (batch, spatial, 1)
        return x

