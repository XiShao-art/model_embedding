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

    cls_df0 = pd.read_pickle('data/classPPIEmbed0.pkl')
    cls_df1 = pd.read_pickle('data/classPPIEmbed1.pkl')
    cls_df2 = pd.read_pickle('data/classPPIEmbed2.pkl')
    cls_df3 = pd.read_pickle('data/classPPIEmbed3.pkl')
    cls_df4 = pd.read_pickle('data/classPPIEmbed4.pkl')
    cls_df5 = pd.read_pickle('data/classPPIEmbed5.pkl')

    embeds_list0 = cls_df0['embeddings'].values
    embeds_list1 = cls_df1['embeddings'].values
    embeds_list2 = cls_df2['embeddings'].values
    embeds_list3 = cls_df3['embeddings'].values
    embeds_list4 = cls_df4['embeddings'].values
    embeds_list5 = cls_df5['embeddings'].values

    classes0 = {v: k for k, v in enumerate(cls_df0['classes'])}
    classes1 = {v: k for k, v in enumerate(cls_df1['classes'])}
    classes2 = {v: k for k, v in enumerate(cls_df2['classes'])}
    classes3 = {v: k for k, v in enumerate(cls_df3['classes'])}
    classes4 = {v: k for k, v in enumerate(cls_df4['classes'])}
    classes5 = {v: k for k, v in enumerate(cls_df5['classes'])}

    test_data = load_data(test_data_file)

    positive = 0
    total = len(test_data)
    random.seed(123)
    for pair in test_data:
        c, d =pair
        classes_num = len(classes0)-1
        dst0 = return_dst(embeds_list0,classes0,c,d)
        dst1 = return_dst(embeds_list1, classes1, c, d)
        dst2 = return_dst(embeds_list2, classes2, c, d)
        dst3 = return_dst(embeds_list3, classes3, c, d)
        dst4 = return_dst(embeds_list4, classes4, c, d)
        dst5 = return_dst(embeds_list5, classes5, c, d)
        dst = 0
        thr = 0.9
        if dst0>thr or dst1>thr or dst2>thr or dst3>thr or dst4>thr or dst5>thr:
            dst = 1


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