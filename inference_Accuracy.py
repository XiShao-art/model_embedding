#!/usr/bin/env python
import random

import click as ck
import numpy
import numpy as np
import pandas as pd
import logging

from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from scipy.stats import rankdata
import random
logging.basicConfig(level=logging.INFO)
epoch = '6000'
@ck.command()


@ck.option(
    '--test-data-file', '-tsdf', default='data/data-test/GALEN_inferences.txt',
    help='')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/classPPIEmbed5.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/relationPPIEmbed.pkl',
    help='Relation embedings file')

def main(  test_data_file,
    cls_embeds_file, rel_embeds_file):
    #embedding_size = 50

    cls_df = pd.read_pickle(cls_embeds_file)

    rel_df = pd.read_pickle(rel_embeds_file)
    nb_classes = len(cls_df)
    nb_relations = len(rel_df)

    embeds_list = cls_df['embeddings'].values
    classes = {v: k for k, v in enumerate(cls_df['classes'])}

    test_data = load_data(test_data_file)

    positive = 0
    total = len(test_data)
    random.seed(123)
    for pair in test_data:
        c, d =pair
        classes_num = len(classes)-1


        # c = np.array(embeds_list[ random.randint(0,classes_num)])
        # d = np.array(embeds_list[ random.randint(0,classes_num)])
        c = np.array(embeds_list[classes[c]])
        d = np.array(embeds_list[classes[d]])
        embedding_dim = int(len(c) / 2)


        c1 = c[:embedding_dim]
        d1 = d[:embedding_dim]

        c2 = np.abs(c[ embedding_dim:])
        d2 = np.abs(d[ embedding_dim:])

        # box

        cr = np.abs(c2)
        dr = np.abs(d2)


        zeros = (np.zeros(d1.shape))

        cen1 = c1
        cen2 = d1
        euc = np.abs(cen1 - cen2)

        dst = np.linalg.norm(np.maximum(euc + cr - dr , zeros))
        if dst<0.9:
            positive+=1


    print(positive/(total+0.0))


def load_data(data_file):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()

            data.append((it[0], it[1]))
    return data

if __name__ == '__main__':
    main()