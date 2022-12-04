import os
import time

import librosa
import numpy as np
import pyworld as pw
import scipy
import torch
from audio import tools
from text import text_to_sequence
from tqdm import tqdm

import preprocess.seminar_utils as utils


def pitch_extract(wave, sampling_rate, pitch_name):
    pitch, t = pw.dio(wave, sampling_rate, frame_period=256000 / sampling_rate)
    pitch = torch.from_numpy(pw.stonemask(wave, pitch, t, sampling_rate))

    idx_inter = torch.where(pitch == 0.)[0]
    idx_data = torch.where(pitch != 0)[0]
    left_bound, right_bound = pitch[idx_data[0]], pitch[idx_data[-1]]

    pitch[idx_inter] = torch.from_numpy(scipy.interpolate.interp1d(idx_data, pitch[idx_data],
                                                                   fill_value=(left_bound, right_bound),
                                                                   assume_sorted=False, bounds_error=False)(
        np.arange(len(pitch)))[idx_inter])

    np.save(pitch_name, pitch)
    return pitch


def energy_extract(wave_name, energy_name):
    mel_spec = tools.get_mel(wave_name)
    spec = tools.inv_mel_spec(mel_spec, None, spec_ret=True)
    energy = torch.linalg.vector_norm(spec, ord=2, dim=1).squeeze()
    np.save(energy_name, energy)
    return energy


def compress(data, duration):
    data_compressed = torch.empty(duration.shape[0])
    start, finish = 0, 0
    for i, data_i in enumerate(duration):
        finish += data_i
        data_compressed[i] = data[start:finish].mean()
        start += data_i
    return torch.nan_to_num(data_compressed, 0.)


def spectro_data_comp(train_config, i, wave_name, duration):
    pitch_compress_name = os.path.join(train_config.compressed_path, "pitch/ljspeech-pitch_comp-%05d.npy" % (i + 1))
    if os.path.isfile(pitch_compress_name):
        pitch_comp = np.load(pitch_compress_name)
        pitch_comp = torch.from_numpy(pitch_comp)
    else:
        pitch_name = os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1))
        if os.path.isfile(pitch_name):
            pitch = np.load(pitch_name)
            pitch = torch.from_numpy(pitch)
        else:
            wave, sampling_rate = librosa.load(wave_name)
            pitch = pitch_extract(wave.astype(np.double), sampling_rate, pitch_name)
        pitch_comp = compress(pitch, duration)
        np.save(pitch_compress_name, pitch_comp)

    energy_compress_name = os.path.join(train_config.compressed_path, "energy/ljspeech-energy_comp-%05d.npy" % (i + 1))
    if os.path.isfile(energy_compress_name):
        energy_comp = np.load(energy_compress_name)
        energy_comp = torch.from_numpy(energy_comp)
    else:
        energy_name = os.path.join(train_config.pitch_path, "ljspeech-energy-%05d.npy" % (i + 1))
        if os.path.isfile(energy_name):
            energy = np.load(energy_name)
            energy = torch.from_numpy(energy)
        else:
            energy = energy_extract(wave_name, energy_name)
        energy_comp = compress(energy, duration)
        np.save(energy_compress_name, energy_comp)

    return energy_comp, pitch_comp


def spectro_data(train_config, i, wave_name):
    wave, sampling_rate = librosa.load(wave_name)

    pitch_name = os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (i + 1))
    if os.path.isfile(pitch_name):
        pitch = np.load(pitch_name)
        pitch = torch.from_numpy(pitch)
    else:
        pitch = pitch_extract(np.array(wave, dtype=np.float64), sampling_rate, pitch_name)

    energy_name = os.path.join(train_config.energy_path, "ljspeech-energy-%05d.npy" % (i + 1))
    if os.path.isfile(energy_name):
        energy = np.load(energy_name)
        energy = torch.from_numpy(energy)
    else:
        energy = energy_extract(wave_name, energy_name)

    return energy, pitch


def predictor_stats(buffer, key, normalize_flag=False):
    buffer_data = [buffer[idx][key] for idx in range(len(buffer))]
    data = torch.concat(buffer_data)
    dist_stats = (torch.mean(data), torch.std(data))

    if normalize_flag:
        for i in tqdm(range(len(buffer_data))):
            buffer[i][key] = (buffer_data[i] - dist_stats[0]) / dist_stats[1]
        buffer_data = [buffer[idx][key] for idx in range(len(buffer))]
        data = torch.concat(buffer_data)
    quant_stats = (torch.min(data), torch.max(data))

    return dist_stats, quant_stats


def get_data_to_buffer(train_config, model_config):
    buffer = list()
    text = utils.process_text(train_config.data_path)
    wave_names = sorted(os.listdir(train_config.wave_path))

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i + 1))
        mel_gt_target = np.load(mel_gt_name)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        duration = np.load(os.path.join(
            train_config.alignment_path, str(i) + ".npy"))

        wave_name = os.path.join(train_config.wave_path, wave_names[i])
        energy, pitch = spectro_data_comp(train_config, i, wave_name,
                                          duration) if model_config.compress_flag else spectro_data(train_config, i,
                                                                                                    wave_name)

        character = text[i][0:len(text[i]) - 1]
        character = np.array(text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, 'energy': energy, 'pitch': pitch})
        if i == 2000: break

    energy_dist_stats, energy_quant_stats = predictor_stats(buffer, 'energy', True)
    pitch_dist_stats, pitch_quant_stats = predictor_stats(buffer, 'pitch', True)

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer, energy_dist_stats, energy_quant_stats, pitch_dist_stats, pitch_quant_stats
