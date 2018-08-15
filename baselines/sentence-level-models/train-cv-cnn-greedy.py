'''
Training script with ramdom splitting dev set
'''
__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset, CVDataset, get_cv_dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import json
import math
import random
import os
import copy

def split_test_set(ratio=0.1):
	# split dev set
	with open('%s/test.json' % args.data_dir, 'r') as f:
		test_instances = json.load(f)
	test_size = len(test_instances)
	indices = list(range(test_size))
	random.shuffle(indices)
	dev_size = math.ceil(test_size*ratio)
	dev_instances = [test_instances[i] for i in indices[:dev_size]]
	test_instances = [test_instances[i] for i in indices[dev_size:]]
	with open('%s/dev_split.json' % args.data_dir, 'w') as f:
		json.dump(dev_instances, f)
	with open('%s/test_split.json' % args.data_dir, 'w') as f:
		json.dump(test_instances, f)

def train(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Training
	logging.info(str(args))

	model = Model(args, device, train_dset.rel2id, word_emb=emb_matrix)
	max_dev_f1 = 0.0
	test_result_on_max_dev_f1 = (0.0, 0.0, 0.0)
	for iter in range(niter):
		# print('Iteration %d:' % iter)
		loss = 0.0
		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			loss_batch = model.update(batch)
			loss += loss_batch
		loss /= len(train_dset.batched_data)
		# print('Loss: %f' % loss)
		valid_loss, (dev_prec, dev_recall, dev_f1) = model.eval(dev_dset)
		logging.info('Iteration %d, Train loss %f' % (iter, loss))
		logging.info(
			'Dev loss/Precision/Recall/F1: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(valid_loss, dev_prec, dev_recall,
																				  dev_f1))
		test_loss, (test_prec, test_recall, test_f1) = model.eval(test_dset)
		logging.info(
			'Test loss/Precision/Recall/F1: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(test_loss, test_prec, test_recall,
																				   test_f1))
		if dev_f1 > max_dev_f1:
			max_dev_f1 = dev_f1
			test_result_on_max_dev_f1 = (test_prec, test_recall, test_f1)
		# Dynamic update lr
		model.update_lr(valid_loss)
	logging.info('Max dev F1: %f' % max_dev_f1)
	test_p, test_r, test_f1 = test_result_on_max_dev_f1
	logging.info('Test result on max dev F1 (P,R,F1): {:.6f}\t{:.6f}\t{:.6f}'.format(test_p, test_r, test_f1))
	csv_file.write('{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
		args.dropout, max_dev_f1, test_p, test_r, test_f1
	))
	csv_file.flush()
	logging.info('\n')

	return max_dev_f1, test_result_on_max_dev_f1


def train_random(args):
	# Training
	logging.info(str(args))

	model = Model(args, device, train_dset.rel2id, word_emb=emb_matrix)
	max_dev_f1 = 0.0
	test_result_on_max_dev_f1 = (0.0, 0.0, 0.0)
	for iter in range(niter):
		# print('Iteration %d:' % iter)
		loss = 0.0
		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			loss_batch = model.update(batch)
			loss += loss_batch
		loss /= len(train_dset.batched_data)
		# print('Loss: %f' % loss)
		valid_loss, (dev_prec, dev_recall, dev_f1) = model.eval(dev_dset)
		logging.info('Iteration %d, Train loss %f' % (iter, loss))
		logging.info(
			'Dev loss/Precision/Recall/F1: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(valid_loss, dev_prec, dev_recall,
																				  dev_f1))
		test_loss, (test_prec, test_recall, test_f1) = model.eval(test_dset)
		logging.info(
			'Test loss/Precision/Recall/F1: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(test_loss, test_prec, test_recall,
																				   test_f1))
		if dev_f1 > max_dev_f1:
			max_dev_f1 = dev_f1
			test_result_on_max_dev_f1 = (test_prec, test_recall, test_f1)
		# Dynamic update lr
		model.update_lr(valid_loss)
	logging.info('Max dev F1: %f' % max_dev_f1)
	test_p, test_r, test_f1 = test_result_on_max_dev_f1
	logging.info('Test result on max dev F1 (P,R,F1): {:.6f}\t{:.6f}\t{:.6f}'.format(test_p, test_r, test_f1))
	csv_file.write('{:.1f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
		args.dropout, max_dev_f1, test_p, test_r, test_f1
	))
	csv_file.flush()
	logging.info('\n')

	return max_dev_f1, test_result_on_max_dev_f1



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/json')
	parser.add_argument('--vocab_dir', type=str, default='data/vocab')
	parser.add_argument('--model', type=str, default='bgru', help='Model')
	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
	parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
	parser.add_argument('--hidden', type=int, default=230, help='RNN hidden state size.')
	parser.add_argument('--window_size', type=int, default=3, help='Convolution window size')
	parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
	# parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Bidirectional RNN.' )
	parser.set_defaults(bidirectional=True)
	parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
	# parser.add_argument('--in_drop', type=float, default=0.5, help='Input dropout rate.')
	# parser.add_argument('--intra_drop', type=float, default=0.3, help='Intra-layer dropout rate.')
	# parser.add_argument('--state_drop', type=float, default=0.5, help='RNN state dropout rate.')
	# parser.add_argument('--out_drop', type=float, default=0.7, help='Output dropout rate.')
	# parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
	# parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
	parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
	parser.add_argument('--no-lower', dest='lower', action='store_false')
	parser.set_defaults(lower=False)

	parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
	parser.add_argument('--position_dim', type=int, default=30, help='Position encoding dimension.')

	parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.9)

	parser.add_argument('--repeat', type=int, default=5)
	parser.add_argument('--num_epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--cudaid', type=int, default=0)
	parser.add_argument('--seed', type=int, default=7698)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
	parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
	parser.add_argument('--log', type=str, default='log', help='Write training log to file.')
	parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
	parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
	parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
	parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/%s.txt" % args.log, mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)


	with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx

	emb_file = args.vocab_dir + '/embedding.npy'
	emb_matrix = np.load(emb_file)
	assert emb_matrix.shape[0] == len(vocab)
	assert emb_matrix.shape[1] == args.emb_dim
	args.vocab_size = len(vocab)
	niter = args.num_epoch

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	print('Getting relation to index from original train set......')
	train_filename = '%s/train.json' % args.data_dir
	original_train = Dataset(train_filename, args, word2id, device, shuffle=True)
	rel2id = original_train.rel2id
	print('Reading data......')
	train_filename = '%s/train_split.json' % args.data_dir
	test_filename = '%s/test.json' % args.data_dir
	dev_filename = '%s/dev_split.json' % args.data_dir
	train_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True)
	test_dset = Dataset(test_filename, args, word2id, device, rel2id=rel2id)
	dev_dset = Dataset(dev_filename, args, word2id, device, rel2id=rel2id)


	print('Using device: %s' % device.type)

	csv_file = open("./log/%s.csv" % args.log, 'w')

	best_dev_f1 = 0.0
	best_setting = copy.deepcopy(args)
	best_test_result = (0.0, 0.0, 0.0)
	# in_drop, intra_drop, out_drop = args.in_drop, args.intra_drop, args.out_drop

	for drop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
		args.dropout = drop
		dev_f1, test_result = train(args)
		if dev_f1 > best_dev_f1:
			best_dev_f1 = dev_f1
			best_setting = copy.deepcopy(args)
			best_test_result = test_result
	args = copy.deepcopy(best_setting)

	test_p, test_r, test_f1 = best_test_result
	logging.info('Tuning end.')
	logging.info('Best setting: %s' % str(best_setting))
	logging.info('Best result: {:.6f}\t{:.6f}\t{:.6f}'.format(test_p, test_r, test_f1))


	for runid in range(1, args.repeat + 1):
		print('Start repeating......')
		print('Reading data......')
		train_filename = '%s/train_split.json' % args.data_dir
		test_filename = '%s/test.json' % args.data_dir
		dev_filename = '%s/dev_split.json' % args.data_dir
		train_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True)
		test_dset = Dataset(test_filename, args, word2id, device, rel2id=rel2id)
		dev_dset = Dataset(dev_filename, args, word2id, device, rel2id=rel2id)

		logging.info('Run model %d times......' % runid)
		dev_f1, test_result = train_random(args)
		logging.info('')

	csv_file.close()