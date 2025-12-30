import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path

# --- Model definition (must match training script) ---
class SmallResNet(torch.nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        def block(cin, cout, stride=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(cout),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                torch.nn.BatchNorm2d(cout),
            )
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, base, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(base),
            torch.nn.ReLU(inplace=True),
        )
        self.b1 = block(base, base, 1)
        self.p1 = torch.nn.MaxPool2d(2)
        self.b2 = block(base, base * 2, 1)
        self.down2 = torch.nn.Conv2d(base, base * 2, 1, bias=False)
        self.p2 = torch.nn.MaxPool2d(2)
        self.b3 = block(base * 2, base * 4, 1)
        self.down3 = torch.nn.Conv2d(base * 2, base * 4, 1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(base * 4, 1)
    def forward(self, x):
        x = self.stem(x)
        r = x
        x = self.b1(x)
        x = self.relu(x + r)
        x = self.p1(x)
        r = self.down2(x)
        x = self.b2(x)
        x = self.relu(x + r)
        x = self.p2(x)
        r = self.down3(x)
        x = self.b3(x)
        x = self.relu(x + r)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        logits = self.head(x).squeeze(-1)
        return logits

def preprocess(audio_path, cfg):
    wav, sr = torchaudio.load(audio_path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != cfg['sample_rate']:
        wav = torchaudio.transforms.Resample(sr, cfg['sample_rate'])(wav)
    target_len = int(cfg['sample_rate'] * cfg['clip_seconds'])
    t = wav.size(-1)
    if t > target_len:
        start = (t - target_len) // 2
        wav = wav[:, start : start + target_len]
    elif t < target_len:
        pad = target_len - t
        wav = torch.nn.functional.pad(wav, (0, pad), mode="constant", value=0.0)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg['sample_rate'],
        n_fft=cfg['n_fft'],
        hop_length=cfg['hop_length'],
        win_length=cfg['win_length'],
        f_min=cfg['f_min'],
        f_max=cfg['f_max'],
        n_mels=cfg['n_mels'],
        power=2.0,
        center=True,
        pad_mode="reflect",
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    mel = melspec(wav)
    mel_db = amp_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.unsqueeze(0)  # [1, 1, n_mels, time]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to CNN model .pt file")
    parser.add_argument("--audio", required=True, help="Path to audio file or directory")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    cfg = checkpoint["config"]
    model = SmallResNet(in_ch=1, base=32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    audio_path = Path(args.audio)
    files = [audio_path] if audio_path.is_file() else list(audio_path.glob("*.flac"))
    for f in files:
        x = preprocess(str(f), cfg)
        with torch.no_grad():
            logits = model(x)
            proba = torch.sigmoid(logits).item()
        print(f"{f}: probability_fake={proba:.4f} predicted_label={'fake' if proba>=0.5 else 'real'}")

if __name__ == "__main__":
    main()