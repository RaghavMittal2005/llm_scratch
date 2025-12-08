from __future__ import annotations
from typing import List,Tuple
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class rlhf_ex:
    prompt:str
    chosen:str
    reject:str



def load_ds(split:str="train[:200]")->List[rlhf_ex]:
    items:List[rlhf_ex]=[]
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        for row in ds:
            ch = str(row.get("chosen", "")).strip()
            rej=str(row.get("reject","")).strip()
            if ch and rej:
                items.append(rlhf_ex(prompt="",chosen=ch,reject=rej))


    except:
        items = [
            rlhf_ex("Summarize: Scaling laws for neural language models.",
                        "Scaling laws describe how performance improves predictably as model size, data, and compute increase.",
                        "Scaling laws are when you scale pictures to look bigger."),
            rlhf_ex("Give two uses of attention in transformers.",
                        "It lets the model focus on relevant tokens and enables parallel context integration across positions.",
                        "It remembers all past words exactly without any computation."),
        ]
    return items

        