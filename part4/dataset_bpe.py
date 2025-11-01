from __future__ import annotations
import torch
from pathlib import Path
from typing import Tuple
from tokeniser_bpe import BPETokenizer
from torch.utils.data import Dataset

class BPEByteDataset(Dataset):
    def __init__(self):
        super().__init__()