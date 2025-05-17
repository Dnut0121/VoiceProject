import argparse, os, glob
import faiss, numpy as np, torch, librosa, soundfile as sf
from model import Converter
from vocoder_model import HiFiGANGenerator
from speechbrain.inference import EncoderClassifier

# Constants
SR, HOP, WIN, N_MELS = 16000, 256, 1024, 80

# Load models: converter, vocoder, speaker index, speaker encoder
def load_models(device):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    conv = Converter().to(device)
    conv.load_state_dict(torch.load(os.path.join(base, 'converter.pth'), map_location=device))
    conv.eval()
    voc = HiFiGANGenerator().to(device)
    voc.load_state_dict(torch.load(os.path.join(base, 'vocoder.pth'), map_location=device))
    voc.eval()
    idx = faiss.read_index(os.path.join(base, 'speaker.index'))
    spk = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        run_opts={"device": device}
    )
    return conv, voc, idx, spk

# Convert raw waveform to mel-spectrogram tensor
def wav2mel(wav, sr0):
    if sr0 != SR:
        wav = librosa.resample(wav, orig_sr=sr0, target_sr=SR)
    mel = librosa.feature.melspectrogram(wav, sr=SR, n_fft=WIN, hop_length=HOP, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    return torch.from_numpy(mel_db).unsqueeze(0)

# Main batch inference
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'change')),
                    help='Directory of input audio files to convert')
    parser.add_argument('--output_dir', type=str,
                    default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'change_converted')),
                    help='Directory to save converted audio')
# Reference voice sample for target speaker; default to data/raw/ref.wav
    default_ref = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ref.wav'))
    parser.add_argument('--ref', type=str,
                    default=default_ref,
                    help=f'Reference voice sample for target speaker (default: {default_ref})')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conv, voc, idx, spk = load_models(device)

    # Prepare target speaker embedding
    ref_wav, ref_sr = sf.read(args.ref)
    ref_mono = librosa.to_mono(ref_wav) if ref_wav.ndim > 1 else ref_wav
    ref_tensor = torch.from_numpy(ref_mono).unsqueeze(0).to(device)
    emb_ref = spk.encode_batch(ref_tensor).squeeze(0).cpu().numpy().astype('float32')
    _, I = idx.search(emb_ref[np.newaxis, :], 1)
    proc_files = glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', '*.npz')))
    emb_target = np.load(proc_files[I[0][0]])['emb']
    emb_target = torch.from_numpy(emb_target).unsqueeze(0).to(device)

    # Iterate over all audio in source_dir
    for audio_path in glob.glob(os.path.join(args.source_dir, '*')):
        if not audio_path.lower().endswith(('.wav', '.mp3', '.flac')):
            continue
        wav, sr0 = sf.read(audio_path)
        mel_src = wav2mel(wav, sr0).to(device)
        with torch.no_grad():
            mel_tgt = conv(mel_src, emb_target)
            wav_out = voc(mel_tgt)
        out_audio = wav_out.squeeze().cpu().numpy()
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(args.output_dir, f"{basename}_converted.wav")
        sf.write(out_path, out_audio, SR)
        print(f"Saved converted: {out_path}")

if __name__ == '__main__':
    main()