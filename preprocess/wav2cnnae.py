import wave
import numpy as np
import torch
import os
import glob
from tqdm import tqdm

import sflib.sound.sigproc.spec_image as spec_image
generator = spec_image.SpectrogramImageGenerator(image_width=10,image_shift=5)

import sflib.speech.feature.autoencoder_pytorch.base as base

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
tr0006_18 = base.load(18, 'csj_0006', 'CSJ0006', device=device, map_location='cuda')
ae2 = tr0006_18.autoencoder


CSJDATA = '/mnt/aoni01/db/CSJ/USB/WAV/core'
OUTPUTDIR = '/mnt/aoni04/yaguchi/code/kayanuma/Pitch_aoni04/data/cnnae/spec/width10_shift5'


def main():
    names = os.listdir(CSJDATA)
    names = [name for name in names if name[0]=='A' or name[0]=='S']
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    for name in tqdm(names):
        
        wav_path = os.path.join(CSJDATA, name)
        spec_path =  os.path.join(OUTPUTDIR, name.replace('.wav', '_spec.npy'))
        pow_path =  os.path.join(OUTPUTDIR, name.replace('.wav', '_pow.npy'))
        
        wf = wave.open(wav_path)
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        # pad = np.zeros(int(16000*0.05), np.int16)
        # x = np.concatenate([pad, x, pad])
        
        with torch.no_grad():
            generator = spec_image.SpectrogramImageGenerator()
            result = generator.input_wave(x)

            power = []
            feature = []

            for j in range(len(result)):
                image_in = result[j].reshape(1, 512, 10)
                image_in = torch.tensor(image_in).float().to(device)

                x, l2 = ae2.encode(image_in)
            
                power.append(l2[0].detach().cpu().data.numpy())
                feature.append(x[0].detach().cpu().data.numpy())

        power = np.vstack(power)
        feature = np.vstack(feature)
    
        np.save(spec_path, feature)
        np.save(pow_path, power)


if __name__ == '__main__':
    main()
