import torch
import torch.nn as nn
import torch.nn.functional as F





class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class PermuteLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # assuming input is N, C, H, W
        output = input.permute(0, 2, 3, 1) # N, H, W, C
        output = super().forward(output)
        output = output.permute(0, 3, 1, 2)
        return output.type_as(input)

class Fp32PermuteLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # assuming input is N, C, H, W
        output = input.permute(0, 2, 3, 1) # N, H, W, C
        output = super().forward(output)
        output = output.permute(0, 3, 1, 2)
        return output.type_as(input)

class LayerNormSecDim(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
