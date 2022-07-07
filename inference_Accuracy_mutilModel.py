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
@ck.option(
    '--model_num', '-mn', default=100,
    help='model num')

def main(  test_data_file,
    cls_embeds_file, rel_embeds_file,model_num):
    #embedding_size = 50
    cls_df_list = []
    classes_list = []

    thr = 0.9
    for i in range(0,model_num):
        cls_df_list.append(pd.read_pickle('data/classPPIEmbed'+str(i)+'.pkl'))
        classes_list.append({v: k for k, v in enumerate(cls_df_list[i]['classes'])})

    test_data = load_data(test_data_file)

    positive = 0
    total = len(test_data)
    random.seed(123)

    for pair in test_data:
        dst = 0
        c, d =pair
        for i in range(0, model_num):

            dst0 = return_dst(cls_df_list[i]['embeddings'].values,classes_list[i],c,d)
            if dst0 > thr :
                dst = 1
                break
        # c = np.array(embeds_list[ random.randint(0,classes_num)])
        # d = np.array(embeds_list[ random.randint(0,classes_num)])

        if dst<1:
            positive+=1


    print(positive/(total+0.0))

def return_dst(embeds_list,classes,a,b):
    # c = np.array(embeds_list[classes[a]])
    # d = np.array(embeds_list[classes[b]])
    c = np.array(embeds_list[random.randint(0,2000)])
    d = np.array(embeds_list[random.randint(0,2000)])
    embedding_dim = int(len(c) / 2)

    c1 = c[:embedding_dim]
    d1 = d[:embedding_dim]

    c2 = np.abs(c[embedding_dim:])
    d2 = np.abs(d[embedding_dim:])

    # box

    cr = np.abs(c2)
    dr = np.abs(d2)

    zeros = (np.zeros(d1.shape))

    cen1 = c1
    cen2 = d1
    euc = np.abs(cen1 - cen2)

    dst = np.linalg.norm(np.maximum(euc + cr - dr, zeros))
    return dst
def load_data(data_file):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()

            data.append((it[0], it[1]))
    return data

if __name__ == '__main__':
    main()