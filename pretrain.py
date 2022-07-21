# -*- coding: utf-8 -*-
# @Filename: pretrain
# @Date: 2022-07-11 17:33
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import json
import math, random, sys
import argparse
import rdkit
import rdkit.RDLogger
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from gnn_model.GNN import GNN
from utils import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def group_node_rep(node_rep, batch_idx, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_idx == 1)
        group.append(node_rep[count:count + num])
        count += num
    return group


def train(args, model_list, loader, optimizer_list, device):
    model, motif_model = model_list
    optimizer_model, optimizer_motif = optimizer_list

    model.train()
    motif_model.train()
    word_acc, topo_acc = 0, 0
    for step, batch in enumerate(tqdm(loader, desc='Interaction')):
        batch_size = len(batch)
        graph_batch = moltree_to_graph_data(batch)
        batch_idx = graph_batch.batch.numpy()
        graph_batch = graph_batch.to(device)
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        node_rep = group_node_rep(node_rep, batch_idx, batch_size)
        loss, wacc, tacc = motif_model(batch, node_rep)

        optimizer_model.zero_grad()
        optimizer_motif.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_motif.step()

        word_acc += wacc
        topo_acc += tacc

        if (step + 1) % 20 == 0:
            word_acc = word_acc / 20 * 100
            topo_acc = topo_acc / 20 * 100
            print('Loss: %.1f, Word: %.2f, Topo: %.2f' % (loss, word_acc, topo_acc))
            word_acc, topo_acc = 0, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--model', type=str, default='mgssl')
    parser.add_argument('--dataset', type=str, default='zinc')
    parser.add_argument('--gnn', type=str, default='gin')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--graphpooling', type=str, default='mean')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--dataset_path', type=str, default='datasets/zinc/all.txt')
    parser.add_argument('--vocab', type=str, default='datasets/zinc/clique.txt')
    parser.add_argument('--order', type=str, default='bfs')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--input_model_file', type=str, default='saved_model/init',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='saved_model/pretrain',
                        help='filename to output the pre-trained model')

    args = parser.parse_args()

    # current_path = os.path.dirname(os.path.realpath(__file__))
    # model_config_path = os.path.join(current_path, 'config/model/' + args.model + '.json')
    # dataset_config_path = os.path.join(current_path, 'config/dataset/' + args.dataset + '.json')
    # overall_config_path = os.path.join(current_path, 'config/overall.json')
    #
    # model_config = json.load(open(model_config_path))
    # dataset_config = json.load(open(dataset_config_path))
    # overall_config = json.load(open(overall_config_path))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    dataset = MoleculeDataset(args.dataset_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=lambda x: x, drop_last=True)

    model = GNN(args.n_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout, gnn_type=args.gnn).to(device)

    if not args.input_model_file == "":
        model.load_state_dict(torch.load(args.input_model_file + '.pth'))

    vocab = [x.strip('\r\n') for x in open(args.vocab)]
    vocab = Vocab(vocab)
    motif_model = Motif_Generation(vocab, args.hidden_size, args.latent_size, 3, device, args.order).to(device)

    model_list = [model, motif_model]
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_motif = optim.Adam(motif_model.parameters(), lr=1e-3, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_motif]

    for epoch in range(1, args.epoch + 1):
        print("====epoch " + str(epoch))

        train(args, model_list, loader, optimizer_list, device)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()
