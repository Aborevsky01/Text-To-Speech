import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, postnet, duration_predicted, mel_target, duration_predictor_target,
                energy_predicted, energy_target, pitch_predicted, pitch_target):
        mel_loss = self.mse_loss(mel, mel_target)
        postnet_loss = self.mse_loss(postnet, mel_target)
        energy_loss = self.mse_loss(energy_predicted, energy_target)
        pitch_loss = self.mse_loss(pitch_predicted, pitch_target)

        duration_predictor_loss = self.l1_loss(duration_predicted,
                                               duration_predictor_target.float())

        return mel_loss, postnet_loss, duration_predictor_loss, energy_loss, pitch_loss
