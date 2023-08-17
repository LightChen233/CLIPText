import torch
from model.template.clip_template import ClipTemplate
from torch.nn.functional import binary_cross_entropy_with_logits


class Model(ClipTemplate):
    def predict(self, input_ids, attention_mask):
        # bsz x num_labels
        logits = self(input_ids, attention_mask)['logits'] / self.model.logit_scale.exp()

        # bsz x (num_labels - 1)
        onehot_preds_without_ood = self._decode(logits[:, :-1])
        # bsz x num_labels
        onehot_preds_with_ood = torch.zeros(logits.shape)

        for batch_id, pred in enumerate(onehot_preds_without_ood):
            if torch.any(pred).item():
                onehot_preds_with_ood[batch_id][:-1] = pred
            else:
                onehot_preds_with_ood[batch_id][-1] = 1

        # bsz x num_labels
        return onehot_preds_with_ood

    def _get_loss(self, logits, golds):
        return binary_cross_entropy_with_logits(input=logits[:, :-1] / self.model.logit_scale.exp(),
                                                target=golds[:, :-1])

    def _decode(self, logits: torch.Tensor):
        # bsz x (num_labels - 1)
        return torch.where(logits > self.args.multi_threshold, 1, 0)
