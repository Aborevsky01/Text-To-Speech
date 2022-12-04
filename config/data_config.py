from dataclasses import dataclass


@dataclass
class MelConfig:
    num_mels = 80


@dataclass
class DataConfig:
    energy_mean, energy_std, energy_min, energy_max = 17.0530, 17.3866, -0.9808, 12.3962

    pitch_mean, pitch_std, pitch_min, pitch_max = 178.7606, 89.1084, -2.0061, 6.3268

    def __init__(self, energy_dist_stats=None, energy_quant_stats=None, pitch_dist_stats=None, pitch_quant_stats=None):
        if energy_dist_stats is not None:
            energy_mean = energy_dist_stats[0].item()
            energy_std = energy_dist_stats[1].item()
            energy_min = energy_quant_stats[0].item()
            energy_max = energy_quant_stats[1].item()

            pitch_mean = pitch_dist_stats[0].item()
            pitch_std = pitch_dist_stats[1].item()
            pitch_min = pitch_quant_stats[0].item()
            pitch_max = pitch_quant_stats[1].item()
