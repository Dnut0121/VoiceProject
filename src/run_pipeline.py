import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from processing.preprocess import preprocess_file, RAW_DIR, PROC_DIR
from train import train
from inference import inference

def main():
    # 전처리
    os.makedirs(PROC_DIR, exist_ok=True)
    for f in os.listdir(RAW_DIR):
        if f.lower().endswith(('.wav','.mp3','.flac')):
            preprocess_file(f)
    print('변환완료')

    # 학습
    train(
        epochs=1,
        batch_size=1,
        steps_per_epoch=1
    )
    print('학습완료')

    # 테스트
    src_wav = os.path.join(PROC_DIR, 'spk01_001.wav')
    tgt_wav = os.path.join(PROC_DIR, 'spk02_001.wav')
    ckpt    = os.path.join(SCRIPT_DIR, 'checkpoints', 'vc_tf_epoch10.h5')
    out     = os.path.join(SCRIPT_DIR, 'out.wav')
    inference(src_wav, tgt_wav, ckpt, out)
    print('✔ Inference done →', out)

if __name__ == '__main__':
    main()
