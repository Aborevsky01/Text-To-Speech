import torch
import torch.nn as nn

import model.auxilary as aux
from model.postnet import CBHG, Linear
from model.seminar_modules import Encoder, Decoder, LengthRegulator, Predictor
from model.var_adaptor import VarianceAdaptor


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config, data_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(model_config, data_config)

        self.duration = Predictor(model_config)
        self.length_regulator = LengthRegulator(model_config)

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

        if model_config.postnet_flag:
            self.postnet_flag = True
            self.postnet = CBHG(mel_config.num_mels, K=8, projections=[256, mel_config.num_mels])
            self.last_linear = Linear(mel_config.num_mels * 2, mel_config.num_mels)
        else:
            self.postnet_flag = False

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~aux.get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, energy_target=None, pitch_target=None,
                mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, beta=1.0, gamma=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        output, energy_out, pitch_out = self.variance_adaptor(x, energy_target, pitch_target, beta, gamma)
        output, duration_predictor_output = self.length_regulator(output, alpha, length_target, mel_max_length)

        if not self.training:
            mel_pos = duration_predictor_output
        output = self.decoder(output, mel_pos)
        output = self.mel_linear(output)

        if self.training:
            output = self.mask_tensor(output, mel_pos, mel_max_length)

        if self.postnet_flag:
            residual = self.postnet(output)
            residual = self.last_linear(residual)
            mel_postnet_output = output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output, mel_pos, mel_max_length)
        else:
            mel_postnet_output = output

        return mel_postnet_output, output, duration_predictor_output, energy_out, pitch_out
