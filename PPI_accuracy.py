#!/usr/bin/env python

import click as ck
import numpy
import numpy as np
import pandas as pd
import logging
import  random
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO)
epoch = '6000'
@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--train-data-file', '-trdf', default='data/data-train/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--valid-data-file', '-vldf', default='data/data-valid/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--test-data-file', '-tsdf', default='data/data-test/4932.protein.links.v10.5.txt',
    help='')
@ck.option(
    '--cls-embeds-file', '-cef', default='data/classPPIEmbed0.pkl',
    help='Class embedings file')
@ck.option(
    '--rel-embeds-file', '-ref', default='data/relationPPIEmbed0.pkl',
    help='Relation embedings file')
@ck.option(
    '--margin', '-m', default=-0.1,
    help='Loss margin')
@ck.option(
    '--params-array-index', '-pai', default=-1,
    help='Params array index')
@ck.option(
    '--model_num', '-mn', default=1,
    help='model num')
#50 0.9702    0.1403
#60 0.96959   0.1358
#70 0.1344
#80 0.9678    0.13243
def main(go_file, train_data_file, valid_data_file, test_data_file,
         cls_embeds_file, rel_embeds_file, margin, params_array_index,model_num):
    thre = 1.1
    prot_embeds_head_list = []
    prot_rs_head_list = []
    rembeds_new_list = []
    negative_flag = False
    for i in range(model_num):
        cls_df_tail = pd.read_pickle('data/classPPIEmbed'+str(i)+'.pkl')
        cls_df_head = pd.read_pickle('data/classPPIEmbed'+str(i)+'.pkl')
        rel_df = pd.read_pickle('data/relationPPIEmbed'+str(i)+'.pkl')
        nb_classes = len(cls_df_head)
        nb_relations = len(rel_df)

        embeds_list_tail = cls_df_tail['embeddings'].values
        classes = {v: k for k, v in enumerate(cls_df_tail['classes'])}
        size = len(embeds_list_tail[0])
        embedding_size=int(size/2)
        embeds_tail = np.zeros((nb_classes, size), dtype=np.float32)
        for i, emb in enumerate(embeds_list_tail):
            embeds_tail[i, :] = emb
        proteins_tail = {}
        for k, v in classes.items():
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
                proteins_tail[k] = v
        prot_index_tail = list(proteins_tail.values())

        #head
        embeds_list_head = cls_df_head['embeddings'].values
        classes = {v: k for k, v in enumerate(cls_df_head['classes'])}
        rembeds_list = rel_df['embeddings'].values
        relations = {v: k for k, v in enumerate(rel_df['relations'])}
        size = len(embeds_list_head[0])
        embeds_head = np.zeros((nb_classes, size), dtype=np.float32)
        for i, emb in enumerate(embeds_list_head):
            embeds_head[i, :] = emb
        proteins_head = {}
        for k, v in classes.items():
            if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
                proteins_head[k] = v
        rs = embeds_head[:, embedding_size:]
        embeds_head = embeds_head[:, :embedding_size]
        prot_index_head = list(proteins_head.values())
        prot_rs_head = rs[prot_index_head, :]
        prot_rs_head_list.append(prot_rs_head)
        prot_embeds_head = embeds_head[prot_index_head, :]
        prot_embeds_head_list.append(prot_embeds_head)
        prot_dict_head = {v: k for k, v in enumerate(prot_index_head)}
        #####################################################################
        # relation
        rsize = len(rembeds_list[0])
        rembeds = np.zeros((nb_relations, rsize), dtype=np.float32)
        for i, emb in enumerate(rembeds_list):
            rembeds[i, :] = emb
        rembeds_new_list.append(rembeds)


    test_data = load_data(test_data_file, classes, relations)
    count = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(test_data)):
        c  = test_data[i][0]
        r  = test_data[i][1]
        d  = test_data[i][2]
        # negative_flag = True
        if random.randint(0,1)==0:
            negative_flag = True
            i -= 1
        else :

            negative_flag = False
        re = 1
        label = 1
        c, r, d = prot_dict_head[classes[c]], relations[r], prot_dict_head[classes[d]]
        if negative_flag:
            label = 0
            c = random.randint(0, len(prot_dict_head) - 1)
            d = random.randint(0, len(prot_dict_head) - 1)


        for i in range(model_num):
            c1 = prot_embeds_head_list[i][c, :].reshape(1, -1)
            c1 = c1+ rembeds_new_list[i][r, :].reshape(1, -1)
            d1 = prot_embeds_head_list[i][d, :].reshape(1, -1)

            c2 = np.abs(prot_rs_head_list[i][c, :].reshape(1, -1))
            d2 = np.abs(prot_rs_head_list[i][d, :].reshape(1, -1))

            # box

            cr = np.abs(c2)
            dr = np.abs(d2)

            zeros = (np.zeros(d1.shape))

            cen1 = c1
            cen2 = d1
            euc = np.abs(cen1 - cen2)

            dst = np.linalg.norm(np.maximum(euc + cr - dr, zeros))
            # print(dst)
            if dst>thre:
                re = 0
        if re==1:
            count+=1

        if label==1:
            if re == 1:
                TP+=1
            else:
                FN+=1
        else:
            if re ==1:
                FP +=1
            else:
                TN+=1
    accu = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    recall=TP/(TP+FN)

    print("accuracy:",accu,", precesion:",prec,", recall:",recall)







def load_data(data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            # data.append((id1, rel, id2))
            data.append((id1, rel, id2))
    return data


def is_inside(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst + rc <= rd


def is_intersect(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    return dst <= rc + rd


def sim(ec, rc, ed, rd):
    dst = np.linalg.norm(ec - ed)
    overlap = max(0, (2 * rc - max(dst + rc - rd, 0)) / (2 * rc))
    edst = max(0, dst - rc - rd)
    res = (overlap + 1 / np.exp(edst)) / 2


if __name__ == '__main__':
    main()
#threshold = 1.4
#model_num 1 : accuracy: 0.9179938530315731 , precesion: 0.8813444810404527 , recall: 0.9670126019273536
#model_num10 : accuracy: 0.9361087827139797 , precesion: 0.9281348397851454 , recall: 0.9438689018647579
#model_num20 : accuracy: 0.9379715004191115 , precesion: 0.9341244627172491 , recall: 0.9409826807228916
#model_num50 : accuracy: 0.9374592530502003 , precesion: 0.942244687821773 , recall: 0.9327279466271312
#model_num100: accuracy: 0.9355965353450685 , precesion: 0.9451207913258513 , recall: 0.9249744019361444

#threshold = 1.3
#model_num 1 : accuracy: 0.9323367793610878 , precesion: 0.9200401167031363 , recall: 0.9460016874472673
#model_num10 : accuracy: 0.9326161870168576 , precesion: 0.9470802919708029 , recall: 0.9167054011341452
#model_num20 : accuracy: 0.9326627549594859 , precesion: 0.9493487698986975 , recall: 0.9144131586283802
#model_num50 : accuracy: 0.9287510477787091 , precesion: 0.9552091554853985 , recall: 0.8999814091838632
#model_num100: accuracy: 0.9302412219428146 , precesion: 0.9596129159672163 , recall: 0.8992319792726936