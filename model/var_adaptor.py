import numpy as np
import torch
import torch.nn as nn

from config import TrainConfig, FastSpeechConfig
from .seminar_modules import Predictor

train_config = TrainConfig()
model_config = FastSpeechConfig()


class VarianceAdaptor(nn.Module):

    def __init__(self, model_config, data_config):
        super(VarianceAdaptor, self).__init__()

        self.energy = Predictor(model_config)
        self.pitch = Predictor(model_config)

        self.energy_embedding = nn.Embedding(256, model_config.encoder_dim)
        self.pitch_embedding = nn.Embedding(256, model_config.encoder_dim)

        self.energy_mean = int(data_config.energy_max + data_config.energy_min) // 2
        self.energy_scale = (data_config.energy_max + abs(data_config.energy_min)) / 256

        self.pitch_mean = int(data_config.pitch_max + data_config.pitch_min) // 2
        self.pitch_scale = (data_config.pitch_max + abs(data_config.pitch_min)) / 256
        self.pitch_logspace = torch.logspace(start=np.log(1),
                                             end=np.log(data_config.pitch_max + abs(data_config.pitch_min) + 1),
                                             steps=255, base=np.exp(1)).to(train_config.device)
        self.data_config = data_config

    def forward(self, x, energy_target, pitch_target, beta=1.0, gamma=1.0):
        pitch_out = self.pitch(x).squeeze()
        energy_out = self.energy(x).squeeze()

        if energy_target is None:
            energy_target = beta * energy_out
            pitch_target = gamma * pitch_out

        energy_quant = torch.quantize_per_tensor(energy_target + abs(self.data_config.energy_min),
                                                 scale=self.energy_scale, zero_point=self.energy_mean,
                                                 dtype=torch.quint8).int_repr().long()
        energy_quant.requires_grad_ = False
        energy_emb = self.energy_embedding(energy_quant)

        if model_config.pitch_log_scale:
            pitch_quant = torch.bucketize(pitch_target + abs(self.data_config.pitch_min) + 1, self.pitch_logspace)
        else:
            pitch_quant = torch.quantize_per_tensor(pitch_target + abs(self.data_config.pitch_min),
                                                    scale=self.pitch_scale, zero_point=self.pitch_mean,
                                                    dtype=torch.quint8).int_repr().long()
        pitch_quant.requires_grad_ = False
        pitch_emb = self.pitch_embedding(pitch_quant)

        x = x + energy_emb + pitch_emb
        return x, energy_out, pitch_out
