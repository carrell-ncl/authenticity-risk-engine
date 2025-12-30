import torch
import torchaudio
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_cnn_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = SmallResNet(in_ch=1, base=32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg

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

def score_cnn_eval_directory(model_path, eval_dir, thresh=0.5, sample_size=None, seed=None):
    """
    Score all (or a sample of) audio files in 'real' and 'fake' subfolders and calculate per-class and overall accuracy.
    Args:
        model_path (str): Path to the trained CNN model (.pt file).
        eval_dir (str): Path to directory containing 'real' and 'fake' subfolders.
        thresh (float, optional): Threshold for classifying as fake. Defaults to 0.5.
        sample_size (int, optional): If set, randomly sample this many files from each class. Defaults to None (use all files).
        seed (int, optional): Random seed for reproducible sampling. Defaults to None.
    Returns:
        dict: Dictionary with per-class and overall accuracy and results.
    """
    import random
    model, cfg = load_cnn_model(model_path)
    results = defaultdict(list)
    y_true_all = []
    y_pred_all = []
    for label_str, label in [('real', 0), ('fake', 1)]:
        class_dir = Path(eval_dir) / label_str
        if not class_dir.is_dir():
            continue
        files = list(class_dir.glob('*.flac'))
        if sample_size is not None and sample_size < len(files):
            rng = random.Random(seed)
            files = rng.sample(files, sample_size)
        for f in files:
            x = preprocess(str(f), cfg)
            with torch.no_grad():
                logits = model(x)
                proba = torch.sigmoid(logits).item()
            pred = int(proba >= thresh)
            results[label_str].append((str(f), pred, proba))
            y_true_all.append(label)
            y_pred_all.append(pred)
    # Calculate accuracy
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    real_mask = y_true_all == 0
    fake_mask = y_true_all == 1
    real_acc = float((y_pred_all[real_mask] == 0).mean()) if real_mask.any() else float('nan')
    fake_acc = float((y_pred_all[fake_mask] == 1).mean()) if fake_mask.any() else float('nan')
    overall_acc = float((y_pred_all == y_true_all).mean()) if y_true_all.size > 0 else float('nan')
    # Precision (positive class=fake)
    tp = int(((y_pred_all == 1) & (y_true_all == 1)).sum())
    fp = int(((y_pred_all == 1) & (y_true_all == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    # Add confusion matrix and labels
    if y_true_all.size > 0 and y_pred_all.size > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        cm_labels = ['real', 'fake']
    else:
        cm = None
        cm_labels = ['real', 'fake']
    return {
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'overall_accuracy': overall_acc,
        'overall_precision': precision,
        'results': dict(results),
        'n_total': int(y_true_all.size),
        'confusion_matrix': cm.tolist() if cm is not None else None,
        'confusion_matrix_labels': cm_labels,
    }

# Example usage:
# results = score_cnn_eval_directory('models/audio_cnn_mel.pt', 'data/audio/eval')
# print(results)
