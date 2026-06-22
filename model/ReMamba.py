import torch
import torch.nn as nn

from .modules import (
    RECGIBlock,
    ReliabilityGuidedHierarchicalDecoder,
    SharedResNet18Encoder,
    TemporalConsistentFeaturePreparation,
)


class ReMambaNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        pretrained=False,
        channels=(64, 128, 256, 512),
        scan_backend="fast",
        ssm_d_state=8,
        ssm_ratio=1.0,
        lambda_reweight=1.0,
    ):
        super().__init__()
        self.encoder = SharedResNet18Encoder(in_channels=in_channels, pretrained=pretrained)
        self.channels = tuple(channels)

        if tuple(self.encoder.channels) != self.channels:
            raise ValueError("Current ReMambaNet implementation expects ResNet18 channels (64,128,256,512).")

        self.prepare = nn.ModuleList(
            [TemporalConsistentFeaturePreparation(c, c) for c in self.channels]
        )
        self.interaction = nn.ModuleList(
            [
                RECGIBlock(
                    c,
                    scan_backend=scan_backend,
                    d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    lambda_reweight=lambda_reweight,
                )
                for c in self.channels
            ]
        )
        self.decoder = ReliabilityGuidedHierarchicalDecoder(self.channels)

    @staticmethod
    def _flatten_features(prepared_1, prepared_2, h1, h2, z, r, u, details, decoder_features):
        features = {}
        for idx in range(len(z)):
            level = idx + 1
            features[f"prep_l{level}_t1"] = prepared_1[idx]
            features[f"prep_l{level}_t2"] = prepared_2[idx]
            features[f"prep_l{level}_diff"] = torch.abs(prepared_1[idx] - prepared_2[idx])

            features[f"recgi_l{level}_filtered_t1"] = h1[idx]
            features[f"recgi_l{level}_filtered_t2"] = h2[idx]
            features[f"recgi_l{level}_filtered_diff"] = torch.abs(h1[idx] - h2[idx])
            features[f"recgi_l{level}_change_z"] = z[idx]
            features[f"recgi_l{level}_reliability"] = r[idx]
            features[f"recgi_l{level}_uncertainty"] = u[idx]

            if details is not None:
                features[f"recgi_l{level}_structural_discrepancy"] = details[idx]["structural_discrepancy"]
                features[f"recgi_l{level}_nuisance_discrepancy"] = details[idx]["nuisance_discrepancy"]
                features[f"recgi_l{level}_reweight"] = details[idx]["reweight"]
                features[f"recgi_l{level}_rmss_t2_to_t1"] = details[idx]["rmss_t2_to_t1"]
                features[f"recgi_l{level}_rmss_t1_to_t2"] = details[idx]["rmss_t1_to_t2"]

            features[f"decoder_l{level}_context_p"] = decoder_features["context"][idx]
            features[f"decoder_l{level}_local_h"] = decoder_features["local_h"][idx]
            features[f"decoder_l{level}_decoded_v"] = decoder_features["decoded_v"][idx]
        return features

    def forward(self, x1, x2, return_features=False):
        raw_1, raw_2 = self.encoder(x1, x2)

        prepared_1 = []
        prepared_2 = []
        for g1, g2, prep in zip(raw_1, raw_2, self.prepare):
            f1, f2 = prep(g1, g2)
            prepared_1.append(f1)
            prepared_2.append(f2)

        h1_list = []
        h2_list = []
        z_list = []
        reliability_maps = []
        uncertainty_maps = []
        details_list = [] if return_features else None

        for f1, f2, block in zip(prepared_1, prepared_2, self.interaction):
            if return_features:
                h1, h2, z, r, u, details = block(f1, f2, return_details=True)
                details_list.append(details)
            else:
                h1, h2, z, r, u = block(f1, f2, return_details=False)
            h1_list.append(h1)
            h2_list.append(h2)
            z_list.append(z)
            reliability_maps.append(r)
            uncertainty_maps.append(u)

        if return_features:
            logits, decoder_features = self.decoder(
                z_list,
                h1_list,
                h2_list,
                output_size=x1.shape[-2:],
                return_features=True,
            )
            features = self._flatten_features(
                prepared_1,
                prepared_2,
                h1_list,
                h2_list,
                z_list,
                reliability_maps,
                uncertainty_maps,
                details_list,
                decoder_features,
            )
            return logits, features

        logits = self.decoder(z_list, h1_list, h2_list, output_size=x1.shape[-2:], return_features=False)
        return {
            "change_pred": logits,
            "reliability_maps": reliability_maps,
            "uncertainty_maps": uncertainty_maps,
        }
