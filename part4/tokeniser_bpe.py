from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
except Exception:
    Tokenizer = None

class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if Tokenizer is None:
            raise ImportError("Please `pip install tokenizers` for BPETokenizer.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, path: Union[str, Path]):
        path = Path(path)
        if path.is_dir():
            files = [str(f) for f in path.glob("*.txt")]  # all .txt in dir
        else:
            files = [str(path)]

        # Create tokenizer with BPE model
        tokenizer = Tokenizer(BPE())
        
        # Add byte-level pre-tokenizer and decoder
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens
        )
        
        # Train
        tokenizer.train(files, trainer)
        self._tok = tokenizer

    def save(self, out_dir: Union[str, Path]):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        assert self._tok is not None, "train tokeniser first"
        
        # Save the tokenizer
        self._tok.save(str(out_dir / "tokeniser.json"))
        
        # Save metadata
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_tokens}
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
    
    def load(self, model_dir: Union[str, Path]):
        dirp = Path(model_dir)
        tokeniser_file = dirp / "tokeniser.json"
        
        if not tokeniser_file.exists():
            raise FileNotFoundError(f"Could not find tokeniser.json in {dirp}")
        
        # Load tokenizer
        self._tok = Tokenizer.from_file(str(tokeniser_file))
        
        # Load metadata if exists
        meta_file = dirp / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)
        
    def encode(self, text: str) -> List[int]:
        assert self._tok is not None, "Tokenizer not loaded or trained"
        ids = self._tok.encode(text).ids
        return ids
    
    def decode(self, ids: List[int]) -> str:
        assert self._tok is not None, "Tokenizer not loaded or trained"
        text = self._tok.decode(ids)
        return text
    
    def __len__(self):
        if self._tok is None:
            return 0
        return self._tok.get_vocab_size()