import os
import json
import clip
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset

prompt_dict = {
    'topic': 'topic: ',
    'emotion': 'interest: ',
    'situation': 'publication: ',
    'agnews': 'type: ',
    'snips': 'clarify: ',
    'trec': 'caption: ',
    'subj': 'match: ',
}


class Dataset(TorchDataset):
    def __init__(self,
                 dataset_name,
                 mode='test',
                 data_dir='./data',
                 model_type='clip',
                 max_input_length=77,
                 text_prompt=False,
                 tokenize=True):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.model_type = model_type
        self.max_input_length = max_input_length

        self.labels, self.desc = read_nli_json(os.path.join(data_dir, dataset_name, f'{dataset_name}-nli.json'))
        self.num_labels = len(self.labels)

        self.prompt = prompt_dict[dataset_name]
        self.text_prompt = text_prompt

        data_path = os.path.join(data_dir, dataset_name, f'{mode}.txt')
        self.data = self._read_data(data_path)

        if tokenize:
            self._tokenize_all()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def _read_data(self, data_path):
        with open(data_path, 'r', encoding="utf8") as file:
            lines = file.readlines()

        data = list()
        for line in lines:
            temp = line.strip('\n').split('\t')
            assert len(temp) == 2

            if self.dataset_name == 'situation':
                label_idx = [self.labels.index(label) for label in temp[0].split()]
            else:
                label_idx = self.labels.index(temp[0])
            input_seq = temp[1]

            data.append({
                'label_idx': label_idx,  # list or int
                'input_seq': input_seq,  # str
            })

        return data

    def _tokenize(self, seq):
        # seq_len
        return clip.tokenize(seq, context_length=self.max_input_length, truncate=True).squeeze(0)

    def _tokenize_all(self):
        for idx in tqdm(range(len(self.data)), desc='converting to features'):
            label_idx = self.data[idx]['label_idx']  # int or list[int]
            input_seq = self.data[idx]['input_seq']

            label_onehot_idx = torch.zeros(self.num_labels)
            label_onehot_idx[label_idx] = 1
            self.data[idx]['label_onehot_idx'] = label_onehot_idx

            if self.text_prompt:
                input_seq = self.prompt + input_seq
            input_ids = self._tokenize(input_seq)
            self.data[idx]['input_ids'] = input_ids


def read_nli_json(filename):
    with open(filename, 'r') as file:
        line = file.readlines()

    labels = json.loads(line[0])
    desc = json.loads(''.join(line[1:]))

    return labels, desc


def collate_fn(batched_data):
    label_list_ids = list()
    label_ids = list()
    input_ids = list()

    for data in batched_data:
        label_list_ids.append(data['label_idx'])
        label_ids.append(data['label_onehot_idx'])  # num_labels
        input_ids.append(data['input_ids'])  # seq_len

    label_ids = torch.stack(label_ids, dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.where(input_ids > 0, 1, 0)
    return {
        'label_list_ids': label_list_ids,  # list(bsz)
        'label_ids': label_ids,  # tensor (bsz, num_labels)
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
