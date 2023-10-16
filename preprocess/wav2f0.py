import os
import sys
import numpy as np
import soundfile as sf
from tqdm import tqdm
from world4py.np import apis

# CSJ
# DATA="/mnt/aoni01/db/CSJ/USB/WAV/core"
# ATR-Trek
# DATA="/mnt/aoni04/yaguchi/code/ResponseTimingEstimator_DA/data/ATR_Annotated/make_annotated_data/wav_mono"
# JSUT
DATA="/mnt/aoni04/yaguchi/code/pitch_konno/data/jsut_ver1.1/basic5000/wav"

# OUTPUT
OUTDIR="/mnt/aoni04/yaguchi/code/pitch/data/JSUT_basic5000/width1_shift1/f0"
# wav2spec.py image_shift
IMAGE_SHIFT=1


def getf0(filename):
    x, fs = sf.read(filename, dtype='float64')
    f0, time_axis = apis.dio(x, fs, frame_period=5.0)
    start_frame = 5
    f0 = f0[start_frame::int(2*IMAGE_SHIFT)]
    f0 = np.float32(f0)
    return f0


def main():
    names = os.listdir(DATA)
    # names = [name for name in names if name[0]=='A' or name[0]=='S']#頭文字AかSのもののみ
    os.makedirs(OUTDIR, exist_ok=True)
    for name in tqdm(names):
        wav_path = os.path.join(DATA, name)
        out_path = os.path.join(OUTDIR, name.replace('.wav', '_f0.npy'))
        frames = sf.info(wav_path).frames
        if frames < int(0.1 * 16000):
            continue
        f0 = getf0(wav_path)
        np.save(out_path, f0)


if __name__ == '__main__':
    main()