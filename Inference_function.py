# written by: Mohammed Salah Al-Radhi <malradhi@tmit.bme.hu>
# at BME-TMIT, Budapest, 28-30 March 2023
# https://github.com/hcy71o/AutoVocoder



from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import json
import os

import numpy as np
import torch

from .complexdataset import MAX_WAV_VALUE
from .env import AttrDict
from .models import Generator, Encoder

h = None
device = 'cpu'
config_file ='/home/malradhi/AutoVocoder/config.json'
checkpoint_encoder_file = '/home/malradhi/AutoVocoder/e_00210000'
checkpoint_generator_file = '/home/malradhi/AutoVocoder/g_00210000'

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict



def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def init():
    with open(config_file) as f:
        data = f.read()
        global h
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)

def inference_test(mel_data):
    init()
    with torch.no_grad():
        encoder = Encoder(h).to(device)
        state_dict_e = load_checkpoint(checkpoint_encoder_file, device)
        encoder.load_state_dict(state_dict_e['encoder'])
        encoder.eval()
        print("Complete loading encoder")

        generator = Generator(h).to(device)
        state_dict_g = load_checkpoint(checkpoint_generator_file, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        print("Complete loading generator")

        x = torch.from_numpy(np.expand_dims(mel_data, axis=0)).to(device).float()
        y_g_hat = generator(x)

        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    return audio

