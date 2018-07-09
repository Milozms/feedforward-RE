'''
Training script for Position-Aware LSTM for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/json')
	parser.add_argument('--vocab_dir', type=str, default='data/vocab')
	parser.add_argument('--model', type=str, default='pa_lstm', help='Model')
	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
	parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
	parser.add_argument('--hidden', type=int, default=200, help='RNN hidden state size.')
	parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
	parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
	# parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
	# parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
	parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
	parser.add_argument('--no-lower', dest='lower', action='store_false')
	parser.set_defaults(lower=False)

	parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
	parser.add_argument('--position_dim', type=int, default=30, help='Position encoding dimension.')

	parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.9)

	parser.add_argument('--num_epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
	parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
	parser.add_argument('--log', type=str, default='log', help='Write training log to file.')
	parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
	parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
	parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
	parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
	args = vars(parser.parse_args())


	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/%s.txt" % args['log'], mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)

	with open(args['vocab_dir'] + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx

	emb_file = args['vocab_dir'] + '/embedding.npy'
	emb_matrix = np.load(emb_file)
	assert emb_matrix.shape[0] == len(vocab)
	assert emb_matrix.shape[1] == args['emb_dim']
	args['vocab_size'] = len(vocab)
	niter = args['num_epoch']

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_dset = Dataset('train', args, word2id, device, shuffle=True)
	dev_dset = Dataset('dev', args, word2id, device, shuffle=False)

	model = Model(args, device, word_emb=emb_matrix)
	print('Using device: %s' % device.type)

	# model.eval(dev_dset)

	# Training
	max_f1 = 0.0
	for iter in range(niter):
		print('Iteration %d:' % iter)
		loss = 0.0
		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			loss_batch = model.update(batch)
			loss += loss_batch
		loss /= len(train_dset.batched_data)
		print('Loss: %f' % loss)
		valid_loss, (prec, recall, f1) = model.eval(dev_dset)
		print('\n')
		if f1 > max_f1:
			max_f1 = f1
			model.save('./save_model/model', iter)
		logging.info('Iteration %d, Train loss %f, Valid loss %f, Precision %f, Recall %f, F1 %f' % (iter, loss, valid_loss, prec, recall, f1))

	print('Max F1: %f' % max_f1)
