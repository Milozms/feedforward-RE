from collections import OrderedDict
from stanza.text.dataset import Dataset
import sys
filename = sys.argv[1]
numproc = int(sys.argv[2])

WORD_FIELD = 'token'
LABEL_FIELD = 'label'
POS_FIELD = 'stanford_pos'
NER_FIELD = 'stanford_ner'
DEPREL_FIELD = 'stanford_deprel'
DEPHEAD_FIELD = 'stanford_head'
SUBJ_FIELD = 'subj'
OBJ_FIELD = 'obj'
# SUBJ_NER_FIELD = 'subj_ner'
# OBJ_NER_FIELD = 'obj_ner'

FIELDS = [WORD_FIELD, LABEL_FIELD, SUBJ_FIELD, OBJ_FIELD, POS_FIELD, NER_FIELD, DEPREL_FIELD, DEPHEAD_FIELD]

odict = OrderedDict()
for field in FIELDS:
	odict[field] = []

for i in range(numproc):
	tmpfile = '%s_%d' % (filename, i)
	tmpd = Dataset.load_conll(tmpfile)
	for row in tmpd:
		for field in FIELDS:
			odict[field].append(row[field])

dataset = Dataset(odict)
dataset.write_conll(filename)