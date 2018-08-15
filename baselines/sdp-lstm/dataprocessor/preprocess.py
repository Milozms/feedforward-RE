import os
import sys
data_dir = sys.argv[1]

def process(fin1, fin2, fout):
    rl = {}
    for line in fin2:
        wl = line.strip().split(' ')
        rl[wl[0]] = int(wl[1])

    for line in fin1:
        wl = line.strip().split('\t')
        if wl[4] not in rl:
            wl[4] = 'None'
        sent = ' '.join(wl[5].split())
        fout.write('0\t'+sent+'\t'+wl[2]+'\t'+wl[3]+'\t'+str(rl[wl[4]])+'\n')


fin1 = open(os.path.join(data_dir, 'original/train.txt'), encoding='utf-8')
fin2 = open(os.path.join(data_dir, 'original/relation2id.txt'), encoding='utf-8')
fout = open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8')

process(fin1, fin2, fout)

fin1 = open(os.path.join(data_dir, 'original/test.txt'), encoding='utf-8')
fin2 = open(os.path.join(data_dir, 'original/relation2id.txt'), encoding='utf-8')
fout = open(os.path.join(data_dir, 'test.txt'), 'w', encoding='utf-8')

process(fin1, fin2, fout)
