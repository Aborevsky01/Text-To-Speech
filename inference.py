#!g1.1
import argparse
import os
import warnings

import text
import torch
import waveglow
from audio import tools

from config import FastSpeechConfig, TrainConfig, MelConfig, DataConfig
from model import FastSpeech
from model.auxilary import get_WaveGlow
from synthesis import synthesis

warnings.filterwarnings('ignore')


def get_data():
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest.",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education.",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
        "I am very happy to see you again!",
        "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old.",
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list


if __name__ == "__main__":
    model_config = FastSpeechConfig()
    train_config = TrainConfig()
    mel_config = MelConfig()
    data_config = DataConfig()

    WaveGlow = get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=int, default=1.0)
    parser.add_argument('--beta', type=int, default=1.0)
    parser.add_argument('--gamma', type=int, default=1.0)
    parser.add_argument('--m_path', type=str, default=None)
    args = parser.parse_args()

    model = FastSpeech(model_config, mel_config, data_config)

    try:
        checkpoint = torch.load(m_path)
        model.load_state_dict(checkpoint['model'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")

    model = model.to(train_config.device)

    model = model.eval()
    data_list = get_data()

    for i, phn in enumerate(data_list):
        mel, mel_cuda = synthesis(model, phn, args.alpha, args.beta, args.gamma)
        print(mel.shape)
        os.makedirs("results", exist_ok=True)

        tools.inv_mel_spec(
            mel, f"results/s={args.alpha}e={args.beta}p={args.gamma}_{i}.wav"
        )

        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f"results/s={args.alpha}e={args.beta}p={args.gamma}_{i}_waveglow.wav"
        )
        break
