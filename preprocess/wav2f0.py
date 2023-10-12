import os
import sys
import numpy as np
from tqdm import tqdm
from world4py.np import apis


CSJDATA="/mnt/aoni01/db/CSJ/USB/WAV/core"
OUTDIR="/mnt/aoni04/yaguchi/code/kayanuma/Pitch_aoni04/data/revised/width1_shift1/f0"
image_shift=1


def getf0(filename):
    x, fs = sf.read(filename, dtype='float64')
    f0, time_axis = apis.dio(x, fs, frame_period=5.0)
    start_frame = 5
    f0 = f0[start_frame::int(2*image_shift)]
    f0 = np.float32(f0)
    return f0


def main():
    names = os.listdir(CSJDATA)
    names = [name for name in names if name[0]=='A' or name[0]=='S']#頭文字AかSのもののみ
    os.makedirs(OUTDIR, exist_ok=True)
    for name in tqdm(names):
        wav_path = os.path.join(CSJDATA, name)
        out_path = os.path.join(OUTDIR, name.replace('.wav', '_f0.npy'))
        frames = sf.info(wav_path).frames
        if frames < int(0.1 * 16000):
            continue
        f0 = getf0(wav_path, out_path=out_path, f0freq=F0FREQ)
        np.save(out_path, f0)


if __name__ == '__main__':
    main()