import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
import sys
import model.utils as utils
import model.noCluster as noCluster
import model.pack as pack

zip = getattr(itertools, 'izip', zip)

dataset = sys.argv[1]
train_file = './data/intermediate/' + dataset + '/rm/train.data'
test_file = './data/intermediate/' + dataset + '/rm/test.data'
feature_file = './data/intermediate/' + dataset + '/rm/feature.txt'
type_file = './data/intermediate/' + dataset + '/rm/type.txt'
none_ind = utils.get_none_id(type_file)
print("None id:", none_ind)
bat_size = 20
embLen = 50

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, embLen)

doc_size, type_size, feature_list, label_list, type_list = utils.load_corpus(train_file)

doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)

nocluster = noCluster.noCluster(embLen, word_size, type_size)

nocluster.load_word_embedding(pos_embedding_tensor)

# nocluster.load_neg_embedding(neg_embedding_tensor)

# optimizer = utils.sgd(nocluster.parameters(), lr=0.025)
optimizer = optim.SGD(nocluster.parameters(), lr=0.025)

torch.cuda.set_device(0)
nocluster.cuda()
if_cuda = True

best_f1 = float('-inf')
best_recall = 0
best_precision = 0
best_meanBestF1 = float('-inf')
packer = pack.repack(0.1, 20, if_cuda)
fl_t, of_t = packer.repack_eva(feature_list_test)

for epoch in range(200):
    print("epoch: " + str(epoch))
    nocluster.train()
    sf_tp, sf_fl = utils.shuffle_data(type_list, feature_list)
    for b_ind in range(0, len(sf_tp), bat_size):
        nocluster.zero_grad()
        if b_ind + bat_size > len(sf_tp):
            b_eind = len(sf_tp)
        else:
            b_eind = b_ind + bat_size
        t_t, fl_rt1, fl_rt2, fl_dt, off_dt = packer.repack(sf_fl[b_ind: b_eind], sf_tp[b_ind: b_eind])
        loss = nocluster.NLL_loss(t_t, fl_rt1, fl_rt2, fl_dt, off_dt, 2)
        loss.backward()
        nn.utils.clip_grad_norm(nocluster.parameters(), 5)
        optimizer.step()
    # evaluation mode
    nocluster.eval()
    scores = nocluster(fl_t, of_t)
    ind = utils.calcInd(scores)
    entropy = utils.calcEntropy(scores)
    f1score, recall, precision, meanBestF1 = utils.CrossValidation(ind.data, entropy.data, label_list_test, none_ind)

    print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
          (f1score,
           recall,
           precision,
           meanBestF1))
    if meanBestF1 > best_meanBestF1:
        best_f1 = f1score
        best_recall = recall
        best_precision = precision
        best_meanBestF1 = meanBestF1

print('Best result: ')
print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
      (best_f1,
       best_recall,
       best_precision,
       best_meanBestF1))
