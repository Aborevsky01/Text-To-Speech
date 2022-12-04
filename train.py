import argparse
import copy
import os

import text
import torch
import torch.nn as nn
from audio import tools
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import FastSpeechConfig, TrainConfig, DataConfig, MelConfig
from loss import FastSpeechLoss
from model import FastSpeech
from preprocess import dataloader
from preprocess.data import get_data_to_buffer
from synthesis import synthesis
from wandb_writer import WanDBWriter


def main(args):
    model_config = FastSpeechConfig()
    train_config = TrainConfig()
    mel_config = MelConfig()

    buffer, energy_dist_stats, energy_quant_stats, pitch_dist_stats, pitch_quant_stats = get_data_to_buffer(
        train_config, model_config)
    data_config = DataConfig(energy_dist_stats, energy_quant_stats, pitch_dist_stats, pitch_quant_stats)
    dataset = dataloader.BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_expand_size,
        shuffle=True,
        collate_fn=dataloader.collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )

    model = FastSpeech(model_config, mel_config, data_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })
    try:
        checkpoint = torch.load(m_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")

    model = model.to(train_config.device)

    fastspeech_loss = FastSpeechLoss()
    current_step = 0
    logger = WanDBWriter(train_config)

    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)
    model.train()

    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            for j, db in enumerate(batchs):
                current_step += 1
                # tqdm_bar.update(1)

                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                energy_target = db["energy"].float().to(train_config.device)
                pitch_target = db["pitch"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward 
                postnet_output, mel_output, duration_predictor_output, energy_predicted, pitch_predicted = model(
                    character,
                    src_pos,
                    energy_target=energy_target,
                    pitch_target=pitch_target,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration)

                # Calc Loss
                mel_loss, postnet_loss, duration_loss, energy_loss, pitch_loss = fastspeech_loss(mel_output,
                                                                                                 postnet_output,
                                                                                                 duration_predictor_output,
                                                                                                 mel_target,
                                                                                                 torch.log(
                                                                                                     1 + duration),
                                                                                                 energy_predicted,
                                                                                                 energy_target,
                                                                                                 pitch_predicted,
                                                                                                 pitch_target)

                total_loss = mel_loss + duration_loss + energy_loss + pitch_loss
                if model_config.postnet_flag: total_loss += postnet_loss

                # Logger

                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()
                e_l = energy_loss.detach().cpu().numpy()
                p_l = pitch_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)
                logger.add_scalar("energy_loss", e_l)
                logger.add_scalar("pitch_loss", p_l)

                if model_config.postnet_flag:
                    ps_l = postnet_loss.detach().cpu().numpy()
                    logger.add_scalar("postnet_loss", ps_l)

                # Backward
                torch.autograd.set_detect_anomaly(False)
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)
        copy_model = copy.deepcopy(model)
        copy_model.eval()

        tests = [
            "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."]
        data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
        mel_inf, mel_cuda = synthesis(copy_model, data_list[0])
        audio_inf = tools.inv_mel_spec(mel_inf, f"results/s={0}_{0}.wav", audio_ret=True)
        logger.add_audio("audio", torch.from_numpy(audio_inf), sample_rate=22050)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
