#!/usr/bin/env python
# evaluate_syntact_cat.py
"""
Evaluate a fine-tuned SpeechBrain emotion-recognition model on the
syntact_cat corpus (train/test CSVs) without data leakage.
"""

# ──────────────────────────────
# Imports
# ──────────────────────────────
import os
import sys
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
from speechbrain.inference import EncoderClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ──────────────────────────────
# Paths
# ──────────────────────────────
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:  # notebook / REPL
    BASE_DIR = Path.cwd()

CSV_DIR      = BASE_DIR / "syntact_cat"
CSV_TRAIN    = CSV_DIR / "db.emotion.categories.train.desired.csv"
CSV_TEST     = CSV_DIR / "db.emotion.categories.test.desired.csv"
CHECKPOINT   = BASE_DIR / "best_fine_tuned_model_state_dict.pt"
PRETRAIN_DIR = BASE_DIR / "pretrained_models" / "emotion_recognition"

# ──────────────────────────────
# Config
# ──────────────────────────────
CFG = dict(
    batch_size =  8,      # bigger is fine for inference
    max_length = 64000,   # 4.0 s at 16 kHz
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# ──────────────────────────────
# Data
# ──────────────────────────────
class EmotionDataset(Dataset):
    """On-disk wave loader with padding / truncation."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_root: Path,
        label_encoder: LabelEncoder,
        max_length: int = 64000,
        target_sr: int = 16_000,
    ):
        self.df             = dataframe
        self.audio_root     = audio_root
        self.le             = label_encoder
        self.max_length     = max_length
        self.target_sr      = target_sr
        self.resampler_cache = {}

    def __len__(self) -> int:
        return len(self.df)

    def _resample(self, wav: torch.Tensor, src_sr: int) -> torch.Tensor:
        if src_sr == self.target_sr:
            return wav
        # Cache one resampler per sample-rate to save instantiation time
        if src_sr not in self.resampler_cache:
            self.resampler_cache[src_sr] = T.Resample(src_sr, self.target_sr)
        return self.resampler_cache[src_sr](wav)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        wav_path = self.audio_root / row["file"]
        waveform, sr = torchaudio.load(wav_path)
        waveform = self._resample(waveform, sr).squeeze(0)  # mono

        # pad / truncate to fixed length
        if waveform.size(0) > self.max_length:
            waveform = waveform[: self.max_length]
        elif waveform.size(0) < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(0)))

        label = self.le.transform([row["emotion"]])[0]
        return waveform, torch.tensor(label, dtype=torch.long)

# ──────────────────────────────
# Helpers
# ──────────────────────────────
@torch.inference_mode()
def validate(
    model: EncoderClassifier,
    dl: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    model.eval()
    total_loss = total_correct = total_samples = 0
    all_preds, all_targets = [], []

    for wav, tgt in dl:
        wav, tgt = wav.to(device), tgt.to(device)

        # SpeechBrain convenience: returns (B, emb_dim)
        emb = model.encode_batch(wav)

        logits = model.mods.output_mlp(emb)
        loss   = criterion(logits, tgt)

        total_loss   += loss.item()
        total_correct += (logits.argmax(1) == tgt).sum().item()
        total_samples += tgt.numel()

        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_targets.extend(tgt.cpu().tolist())

    avg_loss = total_loss / len(dl)
    acc      = total_correct / total_samples
    bca      = balanced_accuracy_score(all_targets, all_preds)
    return avg_loss, acc, bca, all_targets, all_preds

# ──────────────────────────────
# Main
# ──────────────────────────────
def main() -> None:
    # 1 · load CSVs
    df_train = pd.read_csv(CSV_TRAIN)
    df_test  = pd.read_csv(CSV_TEST)

    # 2 · label encoder fitted on **train only**
    le = LabelEncoder().fit(df_train["emotion"])
    num_classes = len(le.classes_)
    print("Label map:", dict(zip(le.classes_, le.transform(le.classes_))))

    # 3 · datasets & loaders
    ds_train = EmotionDataset(df_train, CSV_DIR, le, CFG["max_length"])
    ds_test  = EmotionDataset(df_test , CSV_DIR, le, CFG["max_length"])

    dl_train = DataLoader(ds_train, batch_size=CFG["batch_size"], shuffle=False)
    dl_test  = DataLoader(ds_test , batch_size=CFG["batch_size"], shuffle=False)

    # 4 · model + checkpoint
    model = EncoderClassifier.from_hparams(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        savedir=str(PRETRAIN_DIR),
        run_opts={"device": CFG["device"]},
    )

    in_feats = model.mods.output_mlp.w.in_features
    model.mods.output_mlp = nn.Linear(in_feats, num_classes)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=CFG["device"]))

    model.to(CFG["device"]).eval()
    criterion = nn.CrossEntropyLoss()

    # 5 · evaluation
    tr_loss, tr_acc, tr_bca, _, _ = validate(model, dl_train, criterion, CFG["device"])
    te_loss, te_acc, te_bca, y_true, y_pred = validate(model, dl_test , criterion, CFG["device"])

    print(f"Train  — loss: {tr_loss:.4f} · acc: {tr_acc:.4f} · bca: {tr_bca:.4f}")
    print(f"Test   — loss: {te_loss:.4f} · acc: {te_acc:.4f} · bca: {te_bca:.4f}")

    # 6 · confusion-matrix (optional)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
