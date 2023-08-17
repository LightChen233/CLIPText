import torch
from torch.nn.functional import cross_entropy
from model.template.clip_template import ClipTemplate


class Model(ClipTemplate):
    def _get_loss(self, logits, golds):
        return cross_entropy(input=logits[:, :-1], target=golds[:, :-1])

    def _decode(self, logits):
        # bsz x (num_labels - 1)
        logits = logits[:, :-1]
        probs = torch.softmax(logits, dim=-1)

        # bsz
        max_probs, preds = torch.max(probs, dim=-1)
        preds = torch.where(max_probs >= self.args.threshold, preds, self.num_labels - 1)

        return preds
