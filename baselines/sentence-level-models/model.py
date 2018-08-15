'''
Model wrapper for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler

import utils
from models.position_aware_lstm import PositionAwareLSTM
from models.bgru import BGRU
from models.cnn import CNN
from models.pcnn import PCNN
from models.lstm import LSTM


class Model(object):
	def __init__(self, args, device, rel2id, word_emb=None):
		lr = args.lr
		lr_decay = args.lr_decay
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.max_grad_norm = args.max_grad_norm
		if args.model == 'pa_lstm':
			self.model = PositionAwareLSTM(args, rel2id, word_emb)
		elif args.model == 'bgru':
			self.model = BGRU(args, rel2id, word_emb)
		elif args.model == 'cnn':
			self.model = CNN(args, rel2id, word_emb)
		elif args.model == 'pcnn':
			self.model = PCNN(args, rel2id, word_emb)
		elif args.model == 'lstm':
			self.model = LSTM(args, rel2id, word_emb)
		else:
			raise ValueError
		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss()
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		# self.parameters = self.model.parameters()
		self.optimizer = torch.optim.SGD(self.parameters, lr)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

	def update_lr(self, valid_loss):
		self.scheduler.step(valid_loss)

	def update(self, batch):
		inputs = [p.to(self.device) for p in batch[:-1]]
		labels = batch[-1].to(self.device)
		self.model.train()
		logits = self.model(inputs)
		loss = self.criterion(logits, labels)
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
		self.optimizer.step()
		return loss.item()

	def fix_parameters(self):
		for p in self.model.parameters():
			p.requires_grad = False
		self.model.flinear.bias.requires_grad = True

	def predict(self, batch):
		inputs = [p.to(self.device) for p in batch[:-1]]
		labels = batch[-1].to(self.cpu)
		logits = self.model(inputs).to(self.cpu)
		loss = self.criterion(logits, labels)
		pred = torch.argmax(logits, dim=1).to(self.cpu)
		# corrects = torch.eq(pred, labels)
		# acc_cnt = torch.sum(corrects, dim=-1)
		return logits.tolist(), pred, batch[-1], loss.item()

	def eval(self, dset, vocab=None, output_false_file=None):
		rel_labels = ['']*len(dset.rel2id)
		for label, id in dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0
		for idx, batch in enumerate(tqdm(dset.batched_data)):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b.tolist()
			labels += labels_b.tolist()
			loss += loss_b
			if output_false_file is not None and vocab is not None:
				all_words, pos, ner, subj_pos, obj_pos, labels_ = batch
				all_words = all_words.tolist()
				labels_ = labels_.tolist()
				for i, word_ids in enumerate(all_words):
					if labels[i] != pred[i]:
						length = 0
						for wid in word_ids:
							if wid != utils.PAD_ID:
								length += 1
						words = [vocab[wid] for wid in word_ids[:length]]
						sentence = ' '.join(words)

						subj_words = []
						for sidx in range(length):
							if subj_pos[i][sidx] == 0:
								subj_words.append(words[sidx])
						subj = '_'.join(subj_words)

						obj_words = []
						for oidx in range(length):
							if obj_pos[i][oidx] == 0:
								obj_words.append(words[oidx])
						obj = '_'.join(obj_words)

						output_false_file.write('%s\t%s\t%s\t%s\t%s\n' % (sentence, subj, obj, rel_labels[pred[i]], rel_labels[labels[i]]))

		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels)

	def CrossValidation(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.1, cvnum=100):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(tqdm(test_dset.batched_data)):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b.tolist()
			labels += labels_b.tolist()
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0
		meanBestF1 = 0.0
		pre_ind = utils.calcInd(scores)
		pre_entropy = utils.calcEntropy(scores)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_entropy[ind], labels[ind]] for ind in range(0, len(pre_ind))]
		print('Tuning threshold......')
		for cvind in range(cvnum):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] < threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold
			meanBestF1 += bestF1
			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] < bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1
			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))

		meanBestF1 /= cvnum
		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum
		return loss, f1score, recall, precision, meanBestF1

	def CrossValidation_max_threshold(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.1, cvnum=100):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(tqdm(test_dset.batched_data)):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b.tolist()
			labels += labels_b.tolist()
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0
		meanBestF1 = 0.0
		pre_prob, pre_ind = torch.max(scores, 1)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_prob[ind], labels[ind]] for ind in range(0, len(pre_ind))]
		print('Tuning threshold......')
		for cvind in range(cvnum):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] > threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold
			meanBestF1 += bestF1
			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] > bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1
			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))

		meanBestF1 /= cvnum
		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum
		return loss, f1score, recall, precision, meanBestF1

	def save(self, filename, epoch):
		# params = {
		# 	'model': self.model.state_dict(),
		# 	'config': self.args,
		# 	'epoch': epoch
		# }
		try:
			torch.save(self.model.state_dict(), filename)
			print("Epoch {}, model saved to {}".format(epoch, filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")

	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def load(self, filename):
		params = torch.load(filename)
		self.model.load_state_dict(params)






