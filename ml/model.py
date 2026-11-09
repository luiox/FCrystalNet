"""CrystalNet: the placement heatmap model.

Full-resolution conv stack (the damage field needs per-cell precision, so no
downsampling); the middle block runs dilation=2 to stretch the receptive field to 9 -
the damage tail reaches ~6-8 cells, and a plain 3-stack sees only 7. `factorized=True`
swaps every 3x3x3 for an R(2+1)D-style (1,3,3)+(3,1,1) pair - same field of view,
~45% of the MACs. The aux scalars condition the field two ways: FiLM (multiplicative +
additive modulation after block2, so velocity/armor can reshape the whole field) and a
broadcast add at the bottleneck. Two 1x1 heads: the damage field through Softplus
(non-negative by construction) and the placeability logit kept raw for BCE-with-logits.
The whole point of the exercise: ~12k params, ~1-2 ms forward on one CPU thread.
"""

import torch
import torch.nn as nn


class CrystalNet(nn.Module):

    def __init__(self, channels=11, aux_dim=14, width=24, factorized=False):
        super().__init__()

        def blk(cin, d=1):
            if not factorized:
                return nn.Sequential(
                    nn.Conv3d(cin, width, 3, padding=d, dilation=d),
                    nn.GroupNorm(4, width),
                    nn.SiLU(),
                )
            # R(2+1)D factorization: 3x3x3 -> (1,3,3) spatial + (3,1,1) depth.
            # Same receptive field, ~45% of the MACs; the mid-SiLU keeps the two
            # factors nonlinear so the pair is not just a low-rank conv.
            return nn.Sequential(
                nn.Conv3d(cin, width, (1, 3, 3), padding=(0, d, d), dilation=(1, d, d)),
                nn.SiLU(),
                nn.Conv3d(width, width, (3, 1, 1), padding=(d, 0, 0), dilation=(d, 1, 1)),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            )

        self.stem = blk(channels)
        self.block2 = blk(width, d=2)
        self.block3 = blk(width)
        self.film = nn.Linear(aux_dim, 2 * width)
        self.aux = nn.Sequential(nn.Linear(aux_dim, 32), nn.SiLU(), nn.Linear(32, width))
        self.dmg_head = nn.Sequential(nn.Conv3d(width, 1, 1), nn.Softplus())
        self.valid_head = nn.Conv3d(width, 1, 1)

    def forward(self, x, aux):
        h = self.stem(x)
        h = self.block2(h)
        g, b = self.film(aux).chunk(2, dim=-1)
        h = h * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None]
        h = self.block3(h)
        h = h + self.aux(aux)[:, :, None, None, None]
        return self.dmg_head(h).squeeze(1), self.valid_head(h).squeeze(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
