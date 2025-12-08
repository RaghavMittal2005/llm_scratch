from __future__ import annotations
from typing import List,Tuple
from formatters import Example

class CurriculumLength:
    def __init__(self,items:List[Tuple[str,str]]):
        self._i=0
        self.items = sorted(items, key=lambda p: len(p[0]))

    def __iter__(self):
        self._i=0
        return self
    def __next__(self):
        if self._i>=len(self.items):
            raise StopIteration
        item=self.items[self._i]
        self._i+=1
        return item
        
        
        