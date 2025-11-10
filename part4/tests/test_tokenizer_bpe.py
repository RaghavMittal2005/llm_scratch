import os, tempfile
import pathlib
from pathlib import Path
import sys

from tokeniser_bpe import BPETokenizer

def test_bpe_train_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, 't1.txt')
        with open(txt, 'w') as f:
            f.write('hello hello world')
        tok = BPETokenizer(vocab_size=100)
        tok.train(txt)
        out = os.path.join(d, 'tok')
        tok.save(out)
        tok2 = BPETokenizer()
        tok2.load(out)
        ids = tok2.encode('hello world')
        assert isinstance(ids, list) and len(ids) > 0