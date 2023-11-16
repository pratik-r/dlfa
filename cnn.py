# adapted from code from https://github.com/FrancescoSaverioZuppichini/glasses

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.ops import StochasticDepth

from typing import Optional, List
from math import prod
from copy import deepcopy

import pytorch_lightning as pl

from einops import rearrange

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

torch.use_deterministic_algorithms(True)

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None] * x
        
class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 2,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv1d(
                in_features, in_features, kernel_size=kernel_size, padding='same', bias=False, groups=in_features,
            ),
            # LayerNorm
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv1d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv1d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x = x + res
        return x

class ConvNextStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, ds_kernel_size: int = 2, bn_kernel_size: int = 7, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv1d(in_features, out_features, kernel_size=ds_kernel_size, stride=ds_kernel_size)
            ),
            *[
                BottleNeckBlock(out_features, out_features, kernel_size = bn_kernel_size, **kwargs)
                for _ in range(depth)
            ],
        )
        
class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 4):
        super().__init__(
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=kernel_size),
            nn.GroupNorm(num_groups=1, num_channels=out_features),
        )
        
class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
        stem_kernel_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features, stem_kernel_size)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNextStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0], **kwargs),
                *[
                    ConvNextStage(in_features, out_features, depth, drop_p=drop_p, **kwargs)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
    
class RegressionHead(nn.Sequential):
    def __init__(self, d_out: int):
        super().__init__(
            nn.Flatten(),
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 1),
        )

class InputScaler(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(d_in), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.weight * x

class CNN(pl.LightningModule):
    def __init__(self, d_in, lr, max_epochs, weight_decay, lambda_reg, alpha_reg, drop_path, stem_features, depths, widths, save_init = False, random_state = None, **kwargs):
        super().__init__()
        if random_state is not None:
            torch.manual_seed(random_state)

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

        self.input_scaler = InputScaler(d_in)
        
        self.encoder = ConvNextEncoder(in_channels=1, stem_features=stem_features, depths=depths, widths=widths, drop_p = drop_path, **kwargs)
        
        with torch.no_grad():
            d_out = prod(self.encoder(torch.randn(1,1,d_in)).size())

        self.head = RegressionHead(d_out)
        
        self.init_params = deepcopy(self.state_dict()) if save_init else None

    def forward(self, x):
        x = self.input_scaler(x)
        x = rearrange(x, "b n -> b 1 n")
        x = self.encoder(x)
        x = self.head(x)
        return x
    
    def count_params(self):
        return sum(p.numel() for k,p in self.named_parameters() if p.requires_grad)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()
        loss = F.mse_loss(y, y_hat) + self.lambda_reg * (self.alpha_reg * torch.norm(self.input_scaler.weight, 1) + 0.5 * (1-self.alpha_reg) * torch.square(torch.norm(self.input_scaler.weight, 2)))
        self.log("train_loss", loss.item(), on_step = False, on_epoch = True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return optimizer