# -*- coding: utf-8 -*-
# @Filename: pretrain
# @Date: 2022-07-11 17:33
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import math, random, sys
import argparse
import rdkit
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from gnn_model import GNN
from utils import Vocab, MoleculeDataset, Motif_Generation

def train():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gnn', type=str, default='gin')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nlayer', type=int, default=4)
    parser.add_argument('--embdim', type=int, default=64)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--graphpooling', type=str, default='mean')
    parser.add_argument('--hiddensize', type=int, default=256)
    parser.add_argument('--latentsize', type=int, default=128)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--dataset', type=str, default='datasets/zinc/all.txt')
    parser.add_argument('--vocab', type=str, default='datasets/zinc/clique.txt')
    parser.add_argument('--order', type=str, default='bfs')
    parser.add_argument('--numworkers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--input_model_file', type=str, default='./saved_model/init',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type=str, default='./saved_model/motif_pretrain',
                        help='filename to output the pre-trained model')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x:x, drop_last=True)

    model = GNN(args.num_layer, args.embed_dim, JK=args.JK, drop_ratio=args.drop_ratio, gnn_type=args.gnn_type).to(device)

    if not args.input_model_file == "":
        model.load_state_dict(torch.load(args.input_model_file + '.pth'))

    vocab = [x.strip('\r\n') for x in open(args.vocab)]
    vocab = Vocab(vocab)
    motif_model = Motif_Generation(vocab, args.hidden_size, args.latent_size, 3, device, args.order).to(device)

    model_list = [model, motif_model]
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_motif = optim.Adam(motif_model.parameters(), lr=1e-3, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_motif]

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model_list, loader, optimizer_list, device)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()