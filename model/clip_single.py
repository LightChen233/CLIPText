import torch
from model.template.clip_template import ClipTemplate


class Model(ClipTemplate):
    def _decode(self, logits):
        return torch.argmax(logits, dim=-1)
