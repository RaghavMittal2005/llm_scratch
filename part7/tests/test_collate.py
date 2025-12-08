# Test script
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parent.parent))
from collator import PairCollator
col = PairCollator(bpe_dir="../part4/runs/part4-demo/tokenizer",vocab_size=8000)
test_batch = [("prompt1", "chosen1", "rejected1")]
print("Starting collate...")
result = col.collate(test_batch)
print("Collate finished!")