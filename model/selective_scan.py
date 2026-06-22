import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def _valid_group_count(channels, max_groups=32):
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def cross_scan_2d(x, scans=0):

    b, c, h, w = x.shape
    if scans == 0:
        y = x.new_empty((b, 4, c, h * w))
        y[:, 0] = x.flatten(2, 3)
        y[:, 1] = x.transpose(2, 3).contiguous().flatten(2, 3)
        y[:, 2:4] = torch.flip(y[:, 0:2], dims=[-1])
        return y
    if scans == 1:
        return x.view(b, 1, c, h * w).repeat(1, 4, 1, 1)
    if scans == 2:
        y = x.view(b, 1, c, h * w).repeat(1, 2, 1, 1)
        return torch.cat([y, torch.flip(y, dims=[-1])], dim=1)
    if scans == 3:
        y = x.new_empty((b, 4, c, h * w))
        y[:, 0] = x.flatten(2, 3)
        y[:, 1] = torch.rot90(x, 1, dims=(2, 3)).contiguous().flatten(2, 3)
        y[:, 2] = torch.rot90(x, 2, dims=(2, 3)).contiguous().flatten(2, 3)
        y[:, 3] = torch.rot90(x, 3, dims=(2, 3)).contiguous().flatten(2, 3)
        return y
    raise ValueError(f"Unsupported scan mode: {scans}")


def cross_merge_2d(y, height, width, scans=0):

    b, k, c, h, w = y.shape
    if (h, w) != (height, width):
        raise ValueError(f"Feature size mismatch: got {(h, w)}, expected {(height, width)}")
    y = y.view(b, k, c, -1)
    if scans == 0:
        inv = torch.flip(y[:, 2:4], dims=[-1]).view(b, 2, c, -1)
        y = y[:, 0:2] + inv
        wh = y[:, 1].view(b, c, width, height).transpose(2, 3).contiguous().view(b, c, -1)
        return (y[:, 0] + wh).view(b, c, height, width)
    if scans == 1:
        return y.sum(dim=1).view(b, c, height, width)
    if scans == 2:
        y = y[:, 0:2] + torch.flip(y[:, 2:4], dims=[-1]).view(b, 2, c, -1)
        return y.sum(dim=1).view(b, c, height, width)
    if scans == 3:
        merged = y[:, 0].contiguous().view(b, c, height, width)
        merged = merged + torch.rot90(y[:, 1].view(b, c, width, height), -1, dims=(2, 3))
        merged = merged + torch.rot90(y[:, 2].view(b, c, height, width), -2, dims=(2, 3))
        merged = merged + torch.rot90(y[:, 3].view(b, c, width, height), -3, dims=(2, 3))
        return merged
    raise ValueError(f"Unsupported scan mode: {scans}")


class SelectiveScan2D(nn.Module):

    def __init__(
        self,
        channels,
        d_state=8,
        ssm_ratio=1.0,
        dt_rank="auto",
        d_conv=3,
        dropout=0.0,
        backend="fast",
        scans=0,
        alpha_init=0.5,
        beta_init=0.5,
        eta_init=0.5,
        mu_init=0.5,
        nu_init=0.5,
    ):
        super().__init__()
        if backend not in {"fast", "torch"}:
            raise ValueError("backend must be 'fast' or 'torch'")

        self.channels = int(channels)
        self.d_inner = int(channels * ssm_ratio)
        self.d_state = int(d_state)
        self.dt_rank = math.ceil(channels / 16) if dt_rank == "auto" else int(dt_rank)
        self.k_group = 4
        self.backend = backend
        self.scans = scans

        self.in_proj = nn.Conv2d(self.channels, self.d_inner, kernel_size=1, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Conv1d(
            self.k_group * self.d_inner,
            self.k_group * (self.dt_rank + 2 * self.d_state),
            kernel_size=1,
            groups=self.k_group,
            bias=False,
        )
        self.dt_proj = nn.Conv1d(
            self.k_group * self.dt_rank,
            self.k_group * self.d_inner,
            kernel_size=1,
            groups=self.k_group,
            bias=False,
        )

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).view(1, -1)
        a = a.repeat(self.k_group * self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(a))
        self.A_logs._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
        self.Ds._no_weight_decay = True

        dt = torch.exp(torch.empty(self.k_group, self.d_inner).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(_inverse_softplus(dt))
        self.dt_bias._no_weight_decay = True

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        beta_init = min(max(float(beta_init), 1e-4), 1.0 - 1e-4)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(beta_init / (1.0 - beta_init))))
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        self.nu = nn.Parameter(torch.tensor(float(nu_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def _scan_maps(self, reliability, uncertainty, size, dtype, device):
        h, w = size
        if reliability is None:
            reliability = torch.zeros(1, 1, h, w, dtype=dtype, device=device)
        if uncertainty is None:
            uncertainty = torch.zeros_like(reliability)

        if reliability.shape[-2:] != (h, w):
            reliability = F.interpolate(reliability, size=(h, w), mode="bilinear", align_corners=False)
        if uncertainty.shape[-2:] != (h, w):
            uncertainty = F.interpolate(uncertainty, size=(h, w), mode="bilinear", align_corners=False)

        if reliability.shape[1] != 1:
            reliability = reliability.mean(dim=1, keepdim=True)
        if uncertainty.shape[1] != 1:
            uncertainty = uncertainty.mean(dim=1, keepdim=True)

        reliability = reliability.to(dtype=dtype, device=device)
        uncertainty = uncertainty.to(dtype=dtype, device=device)
        return cross_scan_2d(reliability, scans=self.scans), cross_scan_2d(uncertainty, scans=self.scans)

    def _project_scan_params(self, xs, reliability=None, uncertainty=None, spatial_size=None):
        b, k, c, length = xs.shape
        x_dbl = self.x_proj(xs.view(b, k * c, length)).view(b, k, -1, length)
        dts_rank, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = self.dt_proj(dts_rank.contiguous().view(b, k * self.dt_rank, length))
        dts = dts.view(b, k, c, length)

        if reliability is not None or uncertainty is not None:
            r_seq, u_seq = self._scan_maps(
                reliability,
                uncertainty,
                spatial_size,
                dtype=xs.dtype,
                device=xs.device,
            )
            mod_write = (1.0 + self.alpha * r_seq) * (1.0 - self.beta * u_seq)
            bs = bs * mod_write
            cs = cs * (1.0 + self.eta * r_seq)
            dts = dts + self.mu * r_seq - self.nu * u_seq

        dts = F.softplus(dts + self.dt_bias.view(1, k, c, 1))
        return dts, bs.contiguous(), cs.contiguous()

    def _selective_scan_torch(self, xs, dts, bs, cs):
        b, k, c, length = xs.shape
        n = self.d_state
        a = -self.A_logs.float().exp().view(k, c, n)
        d = self.Ds.float().view(k, c)

        state = xs.new_zeros((b, k, c, n), dtype=torch.float32)
        ys = []
        xs_f = xs.float()
        dts_f = dts.float()
        bs_f = bs.float()
        cs_f = cs.float()

        for t in range(length):
            u_t = xs_f[..., t]
            dt_t = dts_f[..., t]
            b_t = bs_f[..., t]
            c_t = cs_f[..., t]
            delta_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_bu = dt_t.unsqueeze(-1) * b_t[:, :, None, :] * u_t.unsqueeze(-1)
            state = delta_a * state + delta_bu
            y_t = (state * c_t[:, :, None, :]).sum(dim=-1) + d.unsqueeze(0) * u_t
            ys.append(y_t)

        return torch.stack(ys, dim=-1).to(dtype=xs.dtype)

    def _selective_scan_fast(self, xs, dts, bs, cs):
        d = self.Ds.view(1, self.k_group, self.d_inner, 1).to(dtype=xs.dtype, device=xs.device)
        write = 1.0 + torch.tanh(bs.mean(dim=2, keepdim=True))
        read = 1.0 + torch.tanh(cs.mean(dim=2, keepdim=True))
        update = torch.sigmoid(dts) * write * xs
        state = torch.cumsum(update, dim=-1)
        normalizer = torch.cumsum(torch.sigmoid(dts).clamp_min(1e-4), dim=-1)
        state = state / normalizer.clamp_min(1e-4)
        return state * read + xs * d

    def forward(self, x, reliability=None, uncertainty=None):
        b, _, h, w = x.shape
        u = self.act(self.conv2d(self.in_proj(x)))
        xs = cross_scan_2d(u, scans=self.scans)
        dts, bs, cs = self._project_scan_params(xs, reliability, uncertainty, (h, w))

        if self.backend == "torch":
            ys = self._selective_scan_torch(xs, dts, bs, cs)
        else:
            ys = self._selective_scan_fast(xs, dts, bs, cs)

        ys = ys.view(b, self.k_group, self.d_inner, h, w)
        y = cross_merge_2d(ys, h, w, scans=self.scans)
        y = self.out_norm(y)
        y = self.dropout(self.out_proj(y))
        return y


class VSSBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        drop_path=0.0,
        ssm_d_state=8,
        ssm_ratio=1.0,
        mlp_ratio=2.0,
        backend="fast",
    ):
        super().__init__()
        self.norm = LayerNorm2d(hidden_dim)
        self.op = SelectiveScan2D(
            hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            backend=backend,
        )
        self.drop_path = DropPath(drop_path)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm2 = LayerNorm2d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_dim, mlp_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, hidden_dim, kernel_size=1),
        )

    def forward(self, x):
        x = x + self.drop_path(self.op(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
