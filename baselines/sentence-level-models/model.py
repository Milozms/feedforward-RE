'''
Model wrapper for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from models.position_aware_lstm import PositionAwareLSTM
from models.bgru import BGRU


class Model(object):
	def __init__(self, args, device, word_emb=None):
		lr = args['lr']
		lr_decay = args['lr_decay']
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.max_grad_norm = args['max_grad_norm']
		if args['model'] == 'pa_lstm':
			self.model = PositionAwareLSTM(args, word_emb)
		elif args['model'] == 'bgru':
			self.model = BGRU(args, word_emb)
		else:
			raise ValueError
		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss()
		# self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		self.parameters = self.model.parameters()
		self.optimizer = torch.optim.SGD(self.parameters, lr)

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

	def predict(self, batch):
		inputs = [p.to(self.device) for p in batch[:-1]]
		labels = batch[-1].to(self.device)
		logits = self.model(inputs)
		loss = self.criterion(logits, labels)
		pred = torch.argmax(logits, dim=1).to(self.cpu)
		# corrects = torch.eq(pred, labels)
		# acc_cnt = torch.sum(corrects, dim=-1)
		return pred, batch[-1], loss.item()

	def eval(self, dset):
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0
		for idx, batch in enumerate(tqdm(dset.batched_data)):
			pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b.tolist()
			labels += labels_b.tolist()
			loss += loss_b
		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels)

	def save(self, filename, epoch):
		params = {
			'model': self.model.state_dict(),
			'config': self.args,
			'epoch': epoch
		}
		try:
			torch.save(params, filename)
			print("model saved to {}".format(filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")






