import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Evaluator:
    @staticmethod
    def evaluate(onehot_preds, onehot_golds):
        """

        :param onehot_preds: bsz x num_labels
        :param onehot_golds: bsz x num_labels
        :return:
        """

        onehot_golds = onehot_golds.tolist()
        onehot_preds = onehot_preds.tolist()
        return {
            'macro_f1': f1_score(y_true=onehot_golds, y_pred=onehot_preds, average='macro'),
            'micro_f1': f1_score(y_true=onehot_golds, y_pred=onehot_preds, average='micro'),
            'weighted_f1': Evaluator.cal_wf1(onehot_preds, onehot_golds),
            'acc': accuracy_score(y_true=onehot_golds, y_pred=onehot_preds),
        }

    @staticmethod
    def cal_wf1(onehot_preds, onehot_golds):
        num_labels = len(onehot_golds[0])
        onehot_preds = np.array(onehot_preds, dtype=int)
        onehot_golds = np.array(onehot_golds, dtype=int)

        wf1 = 0.0
        tot_weight = 0
        for i in range(num_labels):
            f1 = f1_score(onehot_golds[:, i], onehot_preds[:, i], pos_label=1, average='binary')
            weight = sum(onehot_golds[:, i])
            wf1 += weight * f1
            tot_weight += weight
        wf1 = wf1 / tot_weight
        return wf1
