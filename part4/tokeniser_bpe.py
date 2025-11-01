from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except Exception:
    ByteLevelBPETokenizer = None

class BPETokenizer:
    """Minimal BPE wrapper (HuggingFace tokenizers).
    Trains on a text file or a folder of .txt files. Saves merges/vocab to out_dir.
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please `pip install tokenizers` for BPETokenizer.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self,path:Union[str,Path]):
        path=Path(path)
        if path.is_dir():
            files=[str(f) for f in path.glob("*.txt")]# all .txt in dir #glob yields all files in directory
        else:
            files=[str(path)]

        tokeniser=Tokenizer(ByteLevelBPETokenizer())
        tokeniser.train(files, vocab_size=self.vocab_size,minimum_frequency=2, special_tokens=self.special_tokens)
        self._tok=tokeniser

    def save(self,out_dir:Union[str,Path]):
        out_dir=Path(out_dir)
        out_dir.mkdir(exist_ok=True,parents=True)
        assert self._tok is not None,"train tokeniser first"
        self._tok.save(str(out_dir))
        self._tok.save_model(str(out_dir/"tokeniser.json"))
        meta={"vocab_size":self.vocab_size,"special_tokens":self.special_tokens}
        with open(out_dir/"meta.json","w",encoding="utf-8") as f:
            json.dump(meta,f)
    def load(self,model_dir:Union[str,Path]):
        assert self._tok is None,"tokeniser train"
        dirp=Path(model_dir)
        vocab_file=dirp/"vocab.json"
        merges_file=dirp/"merges.txt"
        tokeniser=dirp/"tokeniser.json"
        if not vocab_file.exists():
            vs = list(dirp.glob("*.json"))
            ms = list(dirp.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab.json/merges.txt in {dirp}")
            vocab = vs[0]
            merges = ms[0]
        # tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        tok = Tokenizer.from_file(str(tokeniser))
        self._tok = tok
        meta_file = dirp / "bpe_meta.json"
        if meta_file.exists():
            with open(meta_file,"r",encoding="utf-8") as f:
                meta=json.load(f)
            self.vocab_size=meta.get("vocab_size",self.vocab_size)
            self.special_tokens=meta.get("special_tokens",self.special_tokens)
        
    def encode(self,text:str)->List[int]:
        id=self._tok.encode(text).ids
        return id
    def decode(self,ids:List[int]):
        text=self._tok.decode(ids)
        return text
    