import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import itertools
import numpy as np
import random
import model.nce as nce
import model.object as obj

class noCluster(nn.Module):
    def __init__(self, emblen, word_size, type_size, if_average=False):
        super(noCluster, self).__init__()
        self.emblen = emblen
        self.word_size = word_size

        self.word_emb = nn.Embedding(word_size, emblen)
        self.word_emb_bag = nn.EmbeddingBag(word_size, emblen)
        self.word_embedding = self.word_emb.weight
        self.word_emb_bag.weight = self.word_embedding

        self.linear = nn.Linear(emblen, type_size, bias=False)
        self.linear.weight.data.zero_()

        self.neg_word = nce.NCE_loss(word_size, emblen)

        self.crit = obj.partCE(if_average=if_average)
        # self.crit = obj.softCE_S(if_average=if_average)
        # self.crit = obj.softCE(if_average=if_average)
        # self.crit = obj.softKL(if_average=if_average)


    def load_word_embedding(self, pre_embeddings):
        self.word_embedding = nn.Parameter(pre_embeddings)
        self.word_emb.weight = self.word_embedding
        self.word_emb_bag.weight = self.word_embedding

    def load_neg_embedding(self, pre_embeddings):
        self.neg_word.load_neg_embedding(pre_embeddings)

    def NLL_loss(self, typeTensor, resampleFeature1, resampleFeature2, feaDrop, offsetDrop, neg_sample):
        scores = self(feaDrop, offsetDrop)
        # batch_size = scores.size(0)
        pos_word = self.word_emb(resampleFeature1)
        loss = self.crit(scores, typeTensor)
        # loss = self.neg_word(pos_word, resampleFeature2, neg_sample, batch_size) + self.crit(scores, typeTensor)
        return loss

    def forward(self, feature_seq, offset_seq):
        men_embedding = self.word_emb_bag(feature_seq, offset_seq)
        return self.linear(F.dropout(men_embedding, p=0.2, training=self.training))
