# pitch

## データ作成
### スペクトログラム
```
python preprocess/wav2spec.py
```
### 基本周波数
```
python preprocess/wav2f0.py
```
### CNN-AE特徴量
```
python preprocess/wav2cnnae.py
```

## モデルの学習（例：LSTM）
```
python scripts/run_pitch_extractor.py configs/config_lstm.json --gpuid 0
```

## 画像生成（例：LSTM）
```
python scripts/run_single_img.py configs/config_lstm.json --gpuid 0
```

## デモ（例：LSTM）
WAV_PATH, IMAGE_WIDTH, IMAGE_SHIFTの設定が必要
```
python demo/run_demo.py configs/config_lstm.json --gpuid 0
```