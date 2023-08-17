import os
import torch
from utils.evaluator import Evaluator
from dataset import Dataset, collate_fn, read_nli_json

from tqdm import tqdm
from torch.utils.data import DataLoader


class Processor:
    def __init__(self, args):
        self.args = args

        json_path = os.path.join(args.data_dir, args.dataset, f'{args.dataset}-nli.json')
        self.labels, self.desc = read_nli_json(json_path)
        self.model = args.Model(args, self.labels, self.desc).to(args.device)

        if args.test:
            self.test_dataloader = self._get_dataloader('test')



    def _get_dataloader(self, mode):
        cache_path = os.path.join(self.args.cache_dir, f'{self.args.name}_{mode}.cache')

        if self.args.use_cache and os.path.exists(cache_path):
            dataset = torch.load(cache_path)
        else:
            dataset = Dataset(dataset_name=self.args.dataset,
                              mode=mode,
                              data_dir=self.args.data_dir,
                              model_type=self.args.model[:4],
                              max_input_length=self.args.max_input_length,
                              text_prompt=self.args.text_prompt)

        if self.args.use_cache and not os.path.exists(cache_path):
            torch.save(dataset, cache_path)

        dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, collate_fn=collate_fn)
        return dataloader

    def eval(self, mode, desc=''):
        self.model.eval()

        onehot_preds_list = list()
        onehot_golds_list = list()
        dataloader = getattr(self, f'{mode}_dataloader')

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'{mode}{desc}'):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)

                intent_preds = self.model.predict(input_ids, attention_mask)

                onehot_preds_list.append(intent_preds)
                onehot_golds_list.append(batch['label_ids'])

        scores = Evaluator.evaluate(onehot_preds=torch.cat(onehot_preds_list, dim=0),
                                    onehot_golds=torch.cat(onehot_golds_list, dim=0))

        print(f'{mode}{desc}:')
        for key, value in scores.items():
            print(f'\t{key}: {value}')

        return scores[self.args.key_score]
