from __future__ import annotations
import argparse, time
import torch
from tokeniser import ByteTokeniser
from dataset import ByteDataset
from model_gpt import GPT
