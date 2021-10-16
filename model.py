from helper import *
from basemodel import BaseModel


class MainModel(BaseModel):
	def __init__(self, arch='bert_base', hf_name='bert-base-uncased', train_batch_size=16, eval_batch_size=16, accumulate_grad_batches=1, learning_rate=1e-5, max_epochs=5,
					optimizer='adamw', adam_epsilon=1e-8, weight_decay=0.0, lr_scheduler='linear_with_warmup', warmup_updates=0.0, freeze_epochs=-1, gpus=1):
		super().__init__(train_batch_size=train_batch_size, max_epochs=max_epochs, gpus=gpus)
		self.save_hyperparameters()

		self.p                         = types.SimpleNamespace()
		self.p.arch                    = arch
		self.p.train_batch_size        = train_batch_size
		self.p.eval_batch_size         = eval_batch_size
		self.p.accumulate_grad_batches = accumulate_grad_batches
		self.p.learning_rate           = learning_rate
		self.p.max_epochs              = max_epochs
		self.p.optimizer               = optimizer
		self.p.adam_epsilon            = adam_epsilon
		self.p.weight_decay            = weight_decay
		self.p.lr_scheduler            = lr_scheduler
		self.p.warmup_updates          = warmup_updates
		self.p.freeze_epochs           = freeze_epochs
		self.p.gpus                    = gpus

		self.text_encoder    = AutoModel.from_pretrained(hf_name)
		out_dim              = self.text_encoder.config.hidden_size
		self.classifier      = nn.Linear(out_dim, 1)

		xavier_normal_(self.classifier.weight)
		self.classifier.bias.data.zero_()

		self.dropout = torch.nn.Dropout(self.text_encoder.config.hidden_dropout_prob)

	def forward(self, batch):
		cls_emb   = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attn_mask'], token_type_ids=batch['type_ids'])['pooler_output']
		cls_emb   = self.dropout(cls_emb)
		logits    = self.classifier(cls_emb)

		return logits

	def calc_loss(self, preds, targets):
		return F.binary_cross_entropy_with_logits(preds, targets)

	def calc_acc(self, preds, targets):
		return 100 * (preds == targets).float().mean()

	def configure_optimizers(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{
				'params'      : [p for n, p in self.text_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.text_encoder.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		optimizer_grouped_parameters += [
			{
				'params'      : [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
				'weight_decay': self.p.weight_decay,
			},
			{
				'params'      : [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
				'weight_decay': 0.0,
			}
		]

		if self.p.optimizer == 'adamw':
			optimizer = AdamW(optimizer_grouped_parameters, lr=self.p.learning_rate, eps=self.p.adam_epsilon, betas=[0.9, 0.98])
		else:
			raise NotImplementedError

		if self.p.lr_scheduler == 'linear_with_warmup':
			if self.p.warmup_updates > 1.0:
				warmup_steps = int(self.p.warmup_updates)
			else:
				warmup_steps = int(self.total_steps * self.p.warmup_updates)
			print(f'\nTotal steps: {self.total_steps} with warmup steps: {warmup_steps}\n')

			scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps)
			scheduler = {
				'scheduler': scheduler,
				'interval': 'step',
				'frequency': 1
			}
		elif self.p.lr_scheduler == 'fixed':
			return [optimizer]
		else:
			raise NotImplementedError

		return [optimizer], [scheduler]
