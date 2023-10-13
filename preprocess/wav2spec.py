import os
import wave
import numpy as np
import spec_image_torch
from tqdm import tqdm
generator = spec_image_torch.SpectrogramImageGeneratorTorch(ramesize=800, frameshift=160, image_width=10, image_shift=5)


CSJDATA = '/mnt/aoni01/db/CSJ/USB/WAV/core'
OUTPUTDIR = '/mnt/aoni04/yaguchi/code/pitch/data/spectrogram_10'


def main():
    names = os.listdir(CSJDATA)
    names = [name for name in names if name[0]=='A' or name[0]=='S']
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    for name in tqdm(names):
        wav_path = os.path.join(CSJDATA, name)
        spec_path =  os.path.join(OUTPUTDIR, name.replace('.wav', '_spec.npy'))
        wf = wave.open(wav_path)
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        # pad = np.zeros(int(16000*0.05), np.int16)
        # x = np.concatenate([pad, x, pad])
        spec = generator.input_wave(x)
        generator.reset()
        np.save(spec_path, spec)

if __name__ == '__main__':
    main()