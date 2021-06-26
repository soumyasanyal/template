from helper import *


class DataModule(pl.LightningDataModule):

	def __init__(self, dataset, train_dataset, dev_dataset, test_dataset, arch, train_batch_size=32, eval_batch_size=32, num_workers=10,):
		super().__init__()
		self.p                  = types.SimpleNamespace()
		self.p.dataset          = dataset
		self.p.train_dataset    = train_dataset		# used in load_dataset()
		self.p.dev_dataset      = dev_dataset		# used in load_dataset()
		self.p.test_dataset     = test_dataset		# used in load_dataset()
		self.p.arch             = arch
		self.p.train_batch_size = train_batch_size
		self.p.eval_batch_size  = eval_batch_size
		self.p.num_workers      = num_workers

	def load_dataset(self, split):
		fnames = [f'../data/processed/{self.p.arch}/{x}/{split}.pkl' for x in getattr(self.p, f'{split}_dataset').split(',')]
		dataset = []
		for fname in fnames:
			with open(fname, 'rb') as f:
				tmp = pickle.load(f)
				dataset = dataset + tmp

		return dataset

	def setup(self, splits='all'):
		self.data = ddict(list)
		if splits == 'all':
			splits = ['train', 'dev', 'test']

		for split in splits:
			self.data[split] = MainDataset(self.load_dataset(split))

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
		parser.add_argument("--dataset", 		 				type=str)
		parser.add_argument("--train_dataset",	default='', 	type=str)
		parser.add_argument("--dev_dataset",	default='', 	type=str)
		parser.add_argument("--test_dataset",	default='', 	type=str)
		parser.add_argument("--num_workers", 	default=10, 	type=int)
		return parser


class MainDataset(Dataset):

	def __init__(self, dataset):
		self.data = dataset

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = {
			'sent': torch.LongTensor(self.data[idx][0]),
			'lbl' : torch.FloatTensor([self.data[idx][1]]),
		}

		return item

	def collater(self, items):
		all_sents = pad_sequence([x['sent'] for x in items], batch_first=True, padding_value=1)
		all_lbls  = torch.cat([x['lbl'] for x in items])

		return all_sents, all_lbls
