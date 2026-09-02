from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class Encoder(nn.Module, ABC):
    input_type: str

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = self.build_encoder()

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def build_encoder(self) -> nn.Module:
        ...