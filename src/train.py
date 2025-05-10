import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import MeanSquaredError
from tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint
from analysis.extract_mel          import extract_mel
from analysis.extract_pitch_harvest import extract_pitch_harvest
from model.advanced_vc_tf_model     import VoiceConversionModelTF
from processing.preprocess          import RAW_DIR, PROC_DIR

HOP_LENGTH = 256
SR         = 16000
N_MELS     = 80

def make_dataset(file_paths, batch_size, steps_per_epoch):
    def gen():
        while True:
            # 랜덤 source/target 선택
            src = random.choice(file_paths)
            tgt = random.choice(file_paths)

            # Mel-spectrogram 추출
            mel_src = extract_mel(src)           # (n_mels, T_mel)
            n_mels, T_mel = mel_src.shape
            times_mel = np.arange(T_mel) * HOP_LENGTH / SR

            # Harvest 피치 추출 및 보간
            f0_t, f0_v = extract_pitch_harvest(src)
            f0_rs = np.interp(times_mel, f0_t, f0_v)

            # 입력 특징: mel + f0 → shape
            feat = np.concatenate([mel_src.T, f0_rs[:, None]], axis=1).astype(np.float32)

            # 타겟 Mel: (T_mel, n_mels)
            mel_tgt = extract_mel(tgt).T.astype(np.float32)

            target_mel = mel_src.T.astype(np.float32)
            yield (feat, mel_tgt), target_mel

    output_signature = (
        (
            tf.TensorSpec(shape=(None, N_MELS+1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, N_MELS),   dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train(
    epochs=10,
    batch_size=4,
    steps_per_epoch=100
):
    os.makedirs('checkpoints', exist_ok=True)

    # processed_audio 폴더에서 .wav 파일 목록 가져오기
    files = [
        os.path.join(PROC_DIR, f)
        for f in os.listdir(PROC_DIR)
        if f.endswith('.wav')
    ]
    if not files:
        raise RuntimeError("Processed audio 파일이 없습니다. preprocess.py를 먼저 실행하세요.")

    # 데이터셋 준비
    ds = make_dataset(files, batch_size, steps_per_epoch)

    # 모델 생성 및 컴파일
    model = VoiceConversionModelTF()
    model.compile(
        optimizer=Adam(2e-4),
        loss=MeanSquaredError()
    )

    tb_cb   = TensorBoard(log_dir='runs/advanced_vc_tf')
    ckpt_cb = ModelCheckpoint(
        filepath='checkpoints/vc_tf_epoch{epoch:02d}.h5',
        save_weights_only=True,
        save_freq='epoch'
    )

    # model.fit 호출
    model.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tb_cb, ckpt_cb],
        verbose=1
    )

if __name__ == '__main__':
    train()
