import os
import wave
import numpy as np
import spec_image_torch
from tqdm import tqdm
generator = spec_image_torch.SpectrogramImageGeneratorTorch(image_width=1,image_shift=1)


# CSJ
# DATA = '/mnt/aoni01/db/CSJ/USB/WAV/core'
# ATR-Trek
# DATA="/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/make_annotated_data/wav_mono"
# JSUT
DATA="/mnt/aoni04/yaguchi/code/pitch_konno/data/jsut_ver1.1/basic5000/wav"

# OUTPUT
OUTPUTDIR = '/mnt/aoni04/yaguchi/code/pitch/data/JSUT_basic5000/width1_shift1/spec'


def main():
    
    names = os.listdir(DATA)
    # names = [name for name in names if name[0]=='A' or name[0]=='S']
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    for name in tqdm(names):
        wav_path = os.path.join(DATA, name)
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