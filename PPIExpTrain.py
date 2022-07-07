#!/usr/bin/env python
import click as ck
import torch.optim as optim


from ELBoxlModel import  ELBoxModel
from utils.elDataLoader import load_data2, load_data
import logging
import torch
logging.basicConfig(level=logging.INFO)

import pandas as pd

@ck.command()
#family_normalized.owl
#yeast-classes-normalized.owl
@ck.option(
    '--data-file', '-df', default='data/data-train/yeast-classes-normalized.owl',
    help='Normalized ontology file (Normalizer.groovy)')
@ck.option(
    '--out-classes-file', '-ocf', default='data/classPPIEmbed5.pkl',
    help='Pandas pkl file with class embeddings')
@ck.option(
    '--out-relations-file', '-orf', default='data/relationPPIEmbed.pkl',
    help='Pandas pkl file with relation embeddings')
@ck.option(
    '--batch-size', '-bs', default=512,
    help='Batch size')
@ck.option(
    '--model_num', '-mn', default=7,
    help='model num')


def main(data_file, out_classes_file, out_relations_file,
         batch_size,model_num):

    device = torch.device('cpu')
    PPI_task = True
    #training procedure
    if PPI_task:
        train_data, classes, relations = load_data2(data_file)
    else:
        train_data, classes, relations = load_data(data_file)


    for i in range(model_num):
        print('model'+str(i)+'#####################')
        embedding_dim = 50
        model = ELBoxModel(device,classes, len(relations), embedding_dim=embedding_dim, batch = batch_size,margin1=-0.05)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = model.to(device)
        train(model,train_data, optimizer)
        model.eval()

       # model = model.to('cuda:0')
        cls_file = 'data/classPPIEmbed'+str(i)+'.pkl'
        df = pd.DataFrame(
            {'classes': list(classes.keys()),
             'embeddings': list(model.classEmbeddingDict.weight.clone().detach().cpu().numpy())})
        df.to_pickle(cls_file)

        rel_file = 'data/relationPPIEmbed'+str(i)+'.pkl'
        df = pd.DataFrame(
            {'relations': list(relations.keys()),
             'embeddings': list(model.relationEmbeddingDict.weight.clone().detach().cpu().numpy())})

        df.to_pickle(rel_file)

def train(model, data, optimizer, num_epochs=8001):
    model.train()

    for epoch in range(num_epochs):
        re = model(data)
        loss = sum(re)
        if epoch%100==0:
            print("epoch:",epoch,'loss:',loss.item(), flush=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # torch.save(model,'./model.pkl')

if __name__ == '__main__':
    main()