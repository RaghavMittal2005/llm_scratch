from __future__ import annotations
from typing import List,Tuple

from dataclasses import dataclass

@dataclass
class Example:
    instruction:str
    response:str

template = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

def format_example(ex:Example)->str:
    return template.format(instruction=ex.instruction.strip(),response=ex.response.strip())

def format_prompt(ex:Example)->str:
    return template.format(instruction=ex.instruction.strip(),response="")
