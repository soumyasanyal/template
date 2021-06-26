import neptune

from helper import *
from data import DataModule
from model import MainModel

model_dict = {
	'model': MainModel,
}

monitor_dict = {
	'model': 'valid_acc_epoch',
}

neptune_api_key = <API_KEY>
neptune_project_name = <PROJECT NAME>

def generate_hydra_overrides():
	# TODO find a better way to override if possible? Maybe we need to use @hydra.main() for this?
	parser = ArgumentParser()
	parser.add_argument('--override')	# Overrides the default hydra config. Setting order is not fixed. E.g., --override rtx_8000,fixed
	args, _ = parser.parse_known_args()
	overrides = []
	if args.override is not None:
		groups = [x for x in os.listdir('./configs/') if os.path.isdir('./configs/' + x)]
		# print(groups)
		for grp in groups:
			confs = [x.replace('.yaml', '') for x in os.listdir('./configs/' + grp) if os.path.isfile('./configs/' + grp + '/' + x)]
			# print(confs)
			for val in args.override.split(','):
				if val in confs:
					overrides.append(f'{grp}={val}')

	return parser, overrides

def load_hydra_cfg(overrides):
	initialize(config_path="./configs/")
	cfg = compose("config", overrides=overrides)
	print('Composed hydra config:\n\n', OmegaConf.to_yaml(cfg))

	return cfg

def parse_args(args=None):
	override_parser, overrides = generate_hydra_overrides()
	hydra_cfg                  = load_hydra_cfg(overrides)
	defaults                   = dict()
	for k,v in hydra_cfg.items():
		if type(v) == DictConfig:
			defaults.update(v)
		else:
			defaults.update({k: v})

	parser = argparse.ArgumentParser(parents=[override_parser], add_help=False)
	parser = pl.Trainer.add_argparse_args(parser)
	parser = model_dict[defaults['model']].add_model_specific_args(parser)
	parser = DataModule.add_data_specific_args(parser)

	parser.add_argument('--seed', 				default=42, 					type=int,)
	parser.add_argument('--name', 				default='test', 				type=str,)
	parser.add_argument('--log_db', 			default='manual_runs', 			type=str,)
	parser.add_argument('--tag_attrs', 			default='model,dataset,arch', 	type=str,)
	parser.add_argument('--ckpt_path', 			default='', 					type=str,)
	parser.add_argument('--eval_splits', 		default='', 					type=str,)
	parser.add_argument('--debug', 				action='store_true')
	parser.add_argument('--offline', 			action='store_true')
	parser.add_argument('--save_checkpoint', 	action='store_true')
	parser.add_argument('--resume_training', 	action='store_true')
	parser.add_argument('--evaluate_ckpt', 		action='store_true')
	parser.set_defaults(**defaults)

	return parser.parse_args()

def get_callbacks(args):

	monitor = monitor_dict[args.model]
	mode = 'max'

	checkpoint_callback = ModelCheckpoint(
		monitor=monitor,
		dirpath=os.path.join(args.root_dir, 'checkpoints'),
		save_top_k=1,
		mode=mode,
		verbose=True,
		save_last=False,
	)

	early_stop_callback = EarlyStopping(
		monitor=monitor,
		min_delta=0.00,
		patience=5,
		verbose=False,
		mode=mode
	)

	return [checkpoint_callback, early_stop_callback]

def get_neptune_logger(args):
	tags		= []
	args_dict	= vars(args)
	args_dict['hostname'] = socket.gethostname()
	for tag_attr in args.tag_attrs.split(','):
		if args_dict.get(tag_attr, None) is not None:
			tags.append(args_dict[tag_attr])
	tags.append(args.log_db)

	neptune_logger = NeptuneLogger(
		api_key=neptune_api_key,
		project_name=neptune_project_name,
		experiment_name=args.name,
		params=args_dict,
		tags=tags,
		offline_mode=args.offline,
	)

	return neptune_logger

def restore_config_params(model, args):
	# TODO Maybe we might require this in future if we want to overwrite some of the ckpt-loaded params
	# restores some of the model args to those of config args

	return model


def main(args, splits='all'):
	pl.seed_everything(args.seed)

	dm = DataModule(
			args.dataset,
			args.train_dataset,
			args.dev_dataset,
			args.test_dataset,
			args.arch,
			train_batch_size=args.train_batch_size,
			eval_batch_size=args.eval_batch_size,
			num_workers=args.num_workers,
		)
	dm.setup(splits=splits)

	print(f'Loading {args.model} - {args.arch} model...')
	model = model_dict[args.model](
			arch=args.arch,
			train_batch_size=args.train_batch_size,
			eval_batch_size=args.eval_batch_size,
			accumulate_grad_batches=args.accumulate_grad_batches,
			learning_rate=args.learning_rate,
			max_epochs=args.max_epochs,
			optimizer=args.optimizer,
			adam_epsilon=args.adam_epsilon,
			weight_decay=args.weight_decay,
			lr_scheduler=args.lr_scheduler,
			warmup_updates=args.warmup_updates,
			freeze_epochs=args.freeze_epochs,
			gpus=args.gpus,
		)

	if args.debug:
		# for DEBUG purposes only
		args.limit_train_batches = 10
		args.limit_val_batches = 10
		args.limit_test_batches = 10
		# args.max_epochs = 1
		# for DEBUG purposes only

	print('Getting Neptune logger...')
	neptune_logger = get_neptune_logger(args)
	# TODO Fix this weird bug: If we don't have this line then experiment_id is None in later lines. Also, this throws some errors
	try:
		print(neptune_logger.experiment)
	except Exception as e:
	   pass
	args.root_dir       = f'../saved/{neptune_logger.experiment_id}' if not args.debug else f'../saved/{args.name}'
	args.neptune_exp_id = neptune_logger.experiment_id
	print(f'Saving to {args.root_dir}')

	print('Building trainer...')
	trainer	= pl.Trainer.from_argparse_args(
		args,
		callbacks=get_callbacks(args),
		logger=neptune_logger,
		num_sanity_val_steps=0,
	)

	return dm, model, trainer


if __name__ == '__main__':
	start_time         = time.time()
	args               = parse_args()
	args.name          = f'{args.model}_{args.dataset}_{args.arch}_{time.strftime("%d_%m_%Y")}_{str(uuid.uuid4())[: 8]}'

	# sanity check
	if args.resume_training:
		assert args.ckpt_path != ''
	if args.evaluate_ckpt:
		assert args.ckpt_path != ''
		assert args.eval_splits != ''

	# Update trainer specific args that are used internally by Trainer (which is initialized from_argparse_args)
	args.precision = 16 if args.fp16 else 32
	if args.resume_training:
		args.resume_from_checkpoint = args.ckpt_path

	# Load the datamodule, model, and trainer used for training (or evaluation)
	if not args.evaluate_ckpt:
		dm, model, trainer = main(args)
	else:
		dm, model, trainer = main(args, splits=args.eval_splits.split(','))

	print(vars(args))

	if not args.evaluate_ckpt:
		# train the model from scratch (or resume training from the checkpoint)
		trainer.fit(model, dm)
		print('Testing the best model...')
		trainer.test(ckpt_path='best')
		if not args.save_checkpoint:
			os.remove(trainer.checkpoint_callback.best_model_path)
	else:
		# evaluate the pretrained model on the provided splits
		model_ckpt       = model.load_from_checkpoint(args.ckpt_path)
		model_ckpt       = restore_config_params(model_ckpt, args)
		print('Testing the best model...')
		for split in args.eval_splits.split(','):
			print(f'Evaluating on split: {split}')
			if split == 'train':
				loader = dm.train_dataloader(shuffle = False)
			elif split == 'dev':
				loader = dm.val_dataloader()
			elif split == 'test':
				loader = dm.test_dataloader()

			trainer.test(model=model_ckpt, test_dataloaders=loader)

	print(f'Time Taken for experiment {args.neptune_exp_id}: {(time.time()-start_time) / 3600}h')
