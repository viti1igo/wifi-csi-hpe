"""Candidate C1: multi-link CNN with one encoder shared by the four links (adapted from Demo4)."""
import torch
from torch import nn
from torch.nn import functional as F


class BinAttention(nn.Module):
    """Reweight the frequency bins from the lag-averaged input (Demo4's SubcarrierSpatialAttention)."""
    def __init__(self, bins):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(bins, 16), nn.GELU(), nn.Linear(16, bins), nn.Sigmoid())

    def forward(self, x):                                   # (B, lags, bins)
        return x * self.net(x.mean(dim=1, keepdim=True))


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(1, 16, k, padding=p) for k, p in (((3, 3), (1, 1)), ((5, 5), (2, 2)), ((7, 3), (3, 1)))])
        self.norm = nn.BatchNorm2d(48)

    def forward(self, x):
        return F.gelu(self.norm(torch.cat([b(x) for b in self.branches], 1)))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                                  nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, 2, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.gelu(self.main(x) + self.skip(x))


class LinkEncoder(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.attention, self.multi = BinAttention(bins), MultiScale()
        self.trunk = nn.Sequential(ResBlock(48, 64), ResBlock(64, 128), nn.AdaptiveAvgPool2d(1))

    def forward(self, x):                                   # (B, lags, bins) -> (B, 128)
        return self.trunk(self.multi(self.attention(x).unsqueeze(1))).flatten(1)


class C1(nn.Module):
    """Input (B, lags, links, bins); output four class logits."""
    def __init__(self, lags=50, links=4, bins=64, dropout=0.2):
        super().__init__()
        self.shape = (lags, links, bins)
        self.encoder = LinkEncoder(bins)                    # the same weights read every link
        self.head = nn.Sequential(nn.Linear(128 * links, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 4))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.shape:
            raise ValueError(f"expected (B, {', '.join(map(str, self.shape))}), got {tuple(x.shape)}")
        b, lags, links, bins = x.shape
        z = self.encoder(x.permute(0, 2, 1, 3).reshape(b * links, lags, bins)).reshape(b, -1)
        return self.head(z)


class Decomposition(nn.Module):
    """Series decomposition (Autoformer, DeepCPD): keep the seasonal part, drop a moving-average trend along tokens."""
    def __init__(self, kernel=5):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):                                   # (B, T, D)
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"), self.kernel, 1).transpose(1, 2)
        return x - trend


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.decomp = Decomposition()

    def forward(self, x):
        h = self.norm1(x)
        x = self.decomp(x + self.attn(h, h, h, need_weights=False)[0])
        return self.decomp(x + self.ffn(self.norm2(x)))


class C2(nn.Module):
    """Simplified DeepCPD: each ACF lag is a token holding all link-bin values; input (B, lags, links, bins)."""
    def __init__(self, lags=50, links=4, bins=64, dropout=0.2, d=64, heads=4, layers=2):
        super().__init__()
        self.shape = (lags, links, bins)
        self.embed = nn.Linear(links * bins, d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, lags, d))
        self.blocks = nn.Sequential(*[Block(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.shape:
            raise ValueError(f"expected (B, {', '.join(map(str, self.shape))}), got {tuple(x.shape)}")
        z = self.blocks(self.embed(x.flatten(2)) + self.pos)
        return self.head(self.norm(z).mean(dim=1))


MODELS = {"c1": C1, "c2": C2}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
