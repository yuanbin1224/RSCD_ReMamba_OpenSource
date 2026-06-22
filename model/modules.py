import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_scan import SelectiveScan2D


def _valid_group_count(channels, max_groups=32):
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, norm=True, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not norm,
            )
        ]
        if norm:
            layers.append(nn.GroupNorm(_valid_group_count(out_channels), out_channels))
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def build_resnet18(pretrained=False, in_channels=3):
    from torchvision.models import resnet18

    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"[ReMamba] Could not load ImageNet ResNet18 weights, using random init: {exc}")
            try:
                base = resnet18(weights=None)
            except TypeError:
                base = resnet18(pretrained=False)
    else:
        try:
            base = resnet18(weights=None)
        except TypeError:
            base = resnet18(pretrained=False)

    if in_channels != 3:
        old = base.conv1
        new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            copied = min(in_channels, 3)
            new.weight[:, :copied].copy_(old.weight[:, :copied])
            if in_channels > copied:
                nn.init.kaiming_normal_(new.weight[:, copied:], mode="fan_out", nonlinearity="relu")
        base.conv1 = new
    return base


class SharedResNet18Encoder(nn.Module):

    channels = (64, 128, 256, 512)

    def __init__(self, in_channels=3, pretrained=False):
        super().__init__()
        base = build_resnet18(pretrained=pretrained, in_channels=in_channels)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])

    def forward(self, x1, x2):
        batch = x1.shape[0]
        x = torch.cat([x1, x2], dim=0)
        x = self.stem(x)
        features_1 = []
        features_2 = []
        for stage in self.stages:
            x = stage(x)
            f1, f2 = torch.split(x, batch, dim=0)
            features_1.append(f1)
            features_2.append(f2)
        return features_1, features_2


class PairwiseTemporalNorm(nn.Module):

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, q1, q2):
        pair = torch.stack([q1, q2], dim=1)
        mean = pair.mean(dim=(1, 3, 4), keepdim=False).view(q1.shape[0], q1.shape[1], 1, 1)
        var = pair.var(dim=(1, 3, 4), unbiased=False, keepdim=False).view(q1.shape[0], q1.shape[1], 1, 1)
        std = torch.sqrt(var + self.eps)
        q1 = self.gamma * (q1 - mean) / std + self.beta
        q2 = self.gamma * (q2 - mean) / std + self.beta
        return q1, q2


class TemporalConsistentFeaturePreparation(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.ptn = PairwiseTemporalNorm(out_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, g1, g2):
        q1 = self.proj(g1)
        q2 = self.proj(g2)
        q1, q2 = self.ptn(q1, q2)
        return q1 + self.refine(q1), q2 + self.refine(q2)


class RECGIBlock(nn.Module):

    def __init__(
        self,
        channels,
        scan_backend="fast",
        d_state=8,
        ssm_ratio=1.0,
        lambda_reweight=1.0,
    ):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.lambda_reweight = float(lambda_reweight)

        self.phi_s = ConvGNAct(channels, channels, kernel_size=1, padding=0)
        self.phi_n = ConvGNAct(channels, channels, kernel_size=1, padding=0)
        self.s2d_struct = SelectiveScan2D(channels, d_state=d_state, ssm_ratio=ssm_ratio, backend=scan_backend)
        self.s2d_nuisance = SelectiveScan2D(channels, d_state=d_state, ssm_ratio=ssm_ratio, backend=scan_backend)

        self.psi_r = nn.Sequential(
            ConvGNAct(channels, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.psi_u = nn.Sequential(
            ConvGNAct(channels, hidden, kernel_size=3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.rm_ssm = SelectiveScan2D(channels, d_state=d_state, ssm_ratio=ssm_ratio, backend=scan_backend)
        self.out_fuse = ConvGNAct(channels * 3, channels, kernel_size=3)

    def forward(self, f1, f2, return_details=False):
        s1, s2 = self.phi_s(f1), self.phi_s(f2)
        n1, n2 = self.phi_n(f1), self.phi_n(f2)

        d_s = self.s2d_struct(torch.abs(s1 - s2))
        d_n = self.s2d_nuisance(torch.abs(n1 - n2))

        r_c = torch.sigmoid(self.psi_r(d_s - d_n))
        u_p = torch.sigmoid(self.psi_u(d_n - d_s))

        weight = 1.0 + self.lambda_reweight * (r_c * (1.0 - u_p)).detach()
        ft1 = f1 * weight
        ft2 = f2 * weight

        rmss_t2_to_t1 = self.rm_ssm(ft2, reliability=r_c, uncertainty=u_p)
        rmss_t1_to_t2 = self.rm_ssm(ft1, reliability=r_c, uncertainty=u_p)
        h1 = ft1 + r_c * rmss_t2_to_t1
        h2 = ft2 + r_c * rmss_t1_to_t2
        z = self.out_fuse(torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1))

        if not return_details:
            return h1, h2, z, r_c, u_p

        details = {
            "structural_discrepancy": d_s,
            "nuisance_discrepancy": d_n,
            "reweight": weight,
            "rmss_t2_to_t1": rmss_t2_to_t1,
            "rmss_t1_to_t2": rmss_t1_to_t2,
            "filtered_t1": h1,
            "filtered_t2": h2,
            "change_z": z,
            "reliability": r_c,
            "uncertainty": u_p,
        }
        return h1, h2, z, r_c, u_p, details


class ReliabilityGuidedHierarchicalDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        self.channels = tuple(channels)
        self.context = nn.ModuleList([ConvGNAct(c * 3, c, kernel_size=3) for c in self.channels])
        self.local_fuse = nn.ModuleList([ConvGNAct(c * 2, c, kernel_size=3) for c in self.channels])
        self.decode_deep = ConvGNAct(self.channels[-1], self.channels[-1], kernel_size=3)
        self.top_down = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            low_c = self.channels[i]
            high_c = self.channels[i + 1]
            self.top_down.append(ConvGNAct(low_c + high_c, low_c, kernel_size=3))
        self.pred = nn.Conv2d(self.channels[0], 1, kernel_size=1)

    def forward(self, z_list, h1_list, h2_list, output_size, return_features=False):
        h_list = []
        p_list = []
        for idx, (z, h1, h2) in enumerate(zip(z_list, h1_list, h2_list)):
            p = self.context[idx](torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1))
            h = self.local_fuse[idx](torch.cat([z, p], dim=1))
            p_list.append(p)
            h_list.append(h)

        v_list = [None] * len(h_list)
        v_list[-1] = self.decode_deep(h_list[-1])
        for idx in range(len(h_list) - 2, -1, -1):
            top = F.interpolate(v_list[idx + 1], size=h_list[idx].shape[-2:], mode="bilinear", align_corners=False)
            v_list[idx] = self.top_down[idx](torch.cat([h_list[idx], top], dim=1))

        logits = self.pred(v_list[0])
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        if return_features:
            return logits, {"context": p_list, "local_h": h_list, "decoded_v": v_list}
        return logits
