"""
model.py
---------
ResNeXt-50 (feature extractor, ImageNet-pretrained) + LSTM (temporal sequence
modelling) for deepfake video classification.

Architecture:
    For each of the `seq_len` face-crop frames in a video:
        frame -> ResNeXt50 (conv layers, classifier head removed) -> 2048-d feature vector
    The sequence of 2048-d vectors -> LSTM -> final hidden state -> FC -> 2-class logits
    (0 = real, 1 = fake)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNeXtLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden=512, lstm_layers=1, bidirectional=False,
                 dropout=0.4, pretrained=True):
        super().__init__()

        resnext = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        )
        # drop the final FC layer, keep everything up to global average pool
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-1])
        self.feature_dim = resnext.fc.in_features  # 2048

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        x: (batch, seq_len, C, H, W)
        """
        batch_size, seq_len, c, h, w = x.shape

        # fold batch and time together to run the CNN once over all frames
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)                # (batch*seq_len, 2048, 1, 1)
        features = features.view(batch_size, seq_len, self.feature_dim)  # (batch, seq_len, 2048)

        lstm_out, (h_n, c_n) = self.lstm(features)
        # use the last time step's output as the video-level representation
        last_out = lstm_out[:, -1, :]

        out = self.dropout(last_out)
        logits = self.classifier(out)
        return logits


def get_model(num_classes=2, pretrained=True, freeze_cnn=False):
    model = ResNeXtLSTM(num_classes=num_classes, pretrained=pretrained)
    if freeze_cnn:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
    return model


if __name__ == "__main__":
    # quick shape sanity check
    m = get_model(pretrained=False)
    dummy = torch.randn(2, 20, 3, 112, 112)  # batch=2, seq_len=20 frames
    out = m(dummy)
    print("Output shape:", out.shape)  # expect (2, 2)
