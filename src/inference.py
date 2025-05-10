import os
import numpy as np
import librosa
import tensorflow as tf
from analysis.extract_mel import extract_mel
from analysis.extract_pitch_harvest import extract_pitch_harvest
from model.advanced_vc_tf_model import VoiceConversionModelTF

def inference(source_path, target_path, checkpoint, output_path):
    model = VoiceConversionModelTF()
    model.load_weights(checkpoint)

    mel_src = extract_mel(source_path)
    _, f0 = extract_pitch_harvest(source_path)
    src_feat = np.stack([mel_src.T, f0], axis=-1)
    mel_tgt = extract_mel(target_path)

    src = np.expand_dims(src_feat.astype(np.float32), 0)
    tgt = np.expand_dims(mel_tgt.T.astype(np.float32), 0)

    out = model(src, tgt, training=False)[0].numpy()
    out_mel = out[..., :80].T
    wav = librosa.feature.inverse.mel_to_audio(
        out_mel, sr=16000, hop_length=256, win_length=1024
    )
    librosa.output.write_wav(output_path, wav, sr=16000)
    print(f"[Inference] {output_path} 생성됨")

if __name__ == '__main__':
    inference(
        source_path=os.path.join(PROC_DIR,'spk01_001.wav'),
        target_path=os.path.join(PROC_DIR,'spk02_001.wav'),
        checkpoint=f"checkpoints/vc_tf_epoch30.h5",
        output_path="out.wav"
    )