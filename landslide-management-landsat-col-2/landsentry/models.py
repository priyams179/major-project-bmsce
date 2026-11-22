"""Bitemporal change-detection networks.

Both models take ``x = [B, 2*C, H, W]`` (pre bands then post bands) and return
per-pixel change logits ``[B, 1, H, W]``.

SiameseCNN
    Shared-weight 4-scale encoder (32/64/128/256 ch). At every scale the two
    dates are fused by absolute difference |f(pre) - f(post)|; the difference
    maps feed a U-Net decoder as skip connections (FC-Siam-diff style).

BITransformer
    BIT-style (Chen et al., 2021, "Remote Sensing Image Change Detection with
    Transformers"): a shared residual CNN backbone to 1/4 resolution, a
    semantic tokenizer that pools each date into L tokens via learned spatial
    attention, a transformer encoder over the 2L bitemporal tokens (context
    exchange between dates), a cross-attention transformer decoder that
    projects the refined tokens back onto the pixel features, and a
    prediction head on |X_pre' - X_post'|.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Siamese CNN
# --------------------------------------------------------------------------- #
def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SiameseCNN(nn.Module):
    def __init__(self, in_bands: int = 4, base: int = 32):
        super().__init__()
        self.in_bands = in_bands
        ch = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([conv_bn_relu(in_bands, ch[0])] +
                                 [conv_bn_relu(ch[i - 1], ch[i]) for i in range(1, 4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2)
                                 for i in range(3, 0, -1)])
        self.dec = nn.ModuleList([conv_bn_relu(ch[i - 1] * 2, ch[i - 1])
                                  for i in range(3, 0, -1)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def encode(self, x):
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats

    def forward(self, x):
        n = self.in_bands
        fa, fb = self.encode(x[:, :n]), self.encode(x[:, n:2 * n])
        diffs = [torch.abs(a - b) for a, b in zip(fa, fb)]
        y = diffs[-1]
        for up, dec, skip in zip(self.up, self.dec, reversed(diffs[:-1])):
            y = dec(torch.cat([up(y), skip], 1))
        return self.head(y)


# --------------------------------------------------------------------------- #
# BIT-style transformer
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(y)) + self.skip(x), inplace=True)


class Backbone(nn.Module):
    """Residual CNN, output stride 4, ``dim`` channels."""

    def __init__(self, in_bands: int, dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_bands, 32, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(ResBlock(32, 48, 2), ResBlock(48, 48))
        self.s2 = nn.Sequential(ResBlock(48, dim, 2), ResBlock(dim, dim))
        self.s3 = nn.Sequential(ResBlock(dim, dim), ResBlock(dim, dim))

    def forward(self, x):
        return self.s3(self.s2(self.s1(self.stem(x))))


class SemanticTokenizer(nn.Module):
    """Pool a feature map into L tokens via L learned spatial softmax maps."""

    def __init__(self, dim: int, n_tokens: int):
        super().__init__()
        self.attn = nn.Conv2d(dim, n_tokens, 1)

    def forward(self, x):                       # x [B, C, H, W]
        a = self.attn(x).flatten(2).softmax(-1)  # [B, L, HW]
        return torch.einsum("bln,bcn->blc", a, x.flatten(2))   # [B, L, C]


class CrossAttnDecoderLayer(nn.Module):
    """Pixels (queries) attend to tokens (keys/values); no pixel self-attn."""

    def __init__(self, dim: int, heads: int, mlp: int, drop: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp, dim), nn.Dropout(drop))

    def forward(self, q, tokens):
        q = q + self.attn(self.n1(q), tokens, tokens, need_weights=False)[0]
        return q + self.mlp(self.n2(q))


class BITransformer(nn.Module):
    def __init__(self, in_bands: int = 4, dim: int = 64, n_tokens: int = 4,
                 enc_depth: int = 1, dec_depth: int = 8, heads: int = 8,
                 drop: float = 0.1):
        super().__init__()
        self.in_bands = in_bands
        self.backbone = Backbone(in_bands, dim)
        self.tokenizer = SemanticTokenizer(dim, n_tokens)
        self.pos = nn.Parameter(torch.randn(1, 2 * n_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, drop, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, enc_depth, enable_nested_tensor=False)
        self.decoder = nn.ModuleList([CrossAttnDecoderLayer(dim, heads, dim * 2, drop)
                                      for _ in range(dec_depth)])
        self.n_tokens = n_tokens
        # Prediction head at full resolution on |X1' - X2'|.
        self.head = nn.Sequential(
            nn.Conv2d(dim, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

    def _decode(self, feat, tokens):
        b, c, h, w = feat.shape
        q = feat.flatten(2).transpose(1, 2)        # [B, HW, C]
        for layer in self.decoder:
            q = layer(q, tokens)
        return q.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        n, H, W = self.in_bands, x.shape[-2], x.shape[-1]
        f1, f2 = self.backbone(x[:, :n]), self.backbone(x[:, n:2 * n])
        t = torch.cat([self.tokenizer(f1), self.tokenizer(f2)], 1) + self.pos
        t = self.encoder(t)
        t1, t2 = t[:, :self.n_tokens], t[:, self.n_tokens:]
        d = torch.abs(self._decode(f1, t1) - self._decode(f2, t2))
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(d)


# --------------------------------------------------------------------------- #
MODELS = {"siamese": SiameseCNN, "transformer": BITransformer}


def build(name: str, in_bands: int = 4) -> nn.Module:
    return MODELS[name](in_bands=in_bands)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
