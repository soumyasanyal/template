from helper import *


class DataModule(pl.LightningDataModule):

	def __init__(self, dataset, train_dataset, dev_dataset, test_dataset, arch, hf_name, train_batch_size=32, eval_batch_size=32, num_workers=10,):
		super().__init__()
		self.p                  = types.SimpleNamespace()
		self.p.dataset          = dataset
		self.p.train_dataset    = train_dataset
		self.p.dev_dataset      = dev_dataset
		self.p.test_dataset     = test_dataset
		self.p.arch             = arch
		self.p.train_batch_size = train_batch_size
		self.p.eval_batch_size  = eval_batch_size
		self.p.num_workers      = num_workers

		self.tokenizer = AutoTokenizer.from_pretrained(hf_name)

	def load_dataset(self, split):
		if split == 'dev':
			split = 'validation'
		data = load_dataset('glue', 'sst2')[split]

		dataset = ddict(list)
		for key in ['sentence', 'label']:
			dataset[key] = data[key]

		return dataset

	def setup(self, splits='all'):
		self.data = ddict(list)
		if splits == 'all':
			splits = ['train', 'dev', 'test']

		for split in splits:
			self.data[split] = MainDataset(self.load_dataset(split), self.tokenizer)

	def train_dataloader(self, shuffle=True):
		return DataLoader(
					self.data['train'],
					batch_size=self.p.train_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['train'].collater,
					shuffle=shuffle,
					pin_memory=True
				)

	def val_dataloader(self):
		return DataLoader(
					self.data['dev'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['dev'].collater,
					pin_memory=True
				)

	def test_dataloader(self):
		return DataLoader(
					self.data['test'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['test'].collater,
					pin_memory=True
				)

	@staticmethod
	def add_data_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument("--dataset", 		 							type=str)
		parser.add_argument("--train_dataset",				default='', 	type=str)
		parser.add_argument("--dev_dataset",				default='', 	type=str)
		parser.add_argument("--test_dataset",				default='', 	type=str)
		parser.add_argument("--num_workers", 				default=10, 	type=int)
		parser.add_argument("--train_batch_size",			default=16,		type=int)
		parser.add_argument("--eval_batch_size", 			default=16, 	type=int)

		return parser


class MainDataset(Dataset):

	def __init__(self, dataset, tokenizer):
		self.data      = dataset
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.data['label'])

	def __getitem__(self, idx):
		item = {
			'sent': self.data['sentence'][idx],
			'lbl' : torch.FloatTensor([self.data['label'][idx]]),
		}

		return item

	def collater(self, items):
		tokenized = self.tokenizer([x['sent'] for x in items], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
		all_lbls  = torch.cat([x['lbl'] for x in items])

		batch = {
			'input_ids': tokenized['input_ids'],
			'attn_mask': tokenized['attention_mask'],
			'type_ids' : tokenized['token_type_ids'],
			'labels'   : all_lbls,
		}

		return batch
