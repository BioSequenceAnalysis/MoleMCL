from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderMaskingPred#, DataListLoader
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, DiscreteGNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd 
from util import MaskAtom
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.autograd import Variable
import timeit

triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
criterion = nn.CrossEntropyLoss()

class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        return x_node, x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    # def loss_tri(self, x, x1, x2):
    #     loss = triplet_loss(x, x1, x2)
    #     return loss

def gen_ran_output(x, edge_index, edge, batch, model, device,grads):
    eta = 6.0
    vice_model = deepcopy(model)
    for (name,vice_param), (name,param) ,(grad)in zip(vice_model.named_parameters(), model.named_parameters(),grads):
        if name.split('.')[0] == 'projection_head':
            vice_param.data = param.data
        else:
           vice_param.data = param.data + eta *grad
    _, z2 = vice_model.forward_cl(x, edge_index, edge, batch)
    return z2

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, epoch, model_list, dataset, optimizer_list, device):

    model, linear_pred_atoms1, linear_pred_bonds1 = model_list
    optimizer_model, optimizer_linear_pred_atoms1, optimizer_linear_pred_bonds1 = optimizer_list
    
    model.train()
    linear_pred_atoms1.train()
    linear_pred_bonds1.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    dataset1 = dataset.shuffle()
    loader1 = DataLoaderMaskingPred(dataset1, batch_size=args.batch_size, shuffle = False, num_workers = args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    epoch_iter = tqdm(loader1, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch1 = batch
        batch1 = batch1.to(device)
        node_rep1, graph_rep1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        with torch.no_grad():
            batch_origin_x = copy.deepcopy(batch1.x)
            batch_origin_x[batch1.masked_atom_indices] = batch1.mask_node_label
            batch_origin_edge = copy.deepcopy(batch1.edge_attr)
            batch_origin_edge[batch1.connected_edge_indices] = batch1.mask_edge_label   
            batch_origin_edge[batch1.connected_edge_indices + 1] = batch1.mask_edge_label         
            atom_ids = batch_origin_x[:,0]
            labels1 = atom_ids[batch1.masked_atom_indices]
            _, graph_rep = model.forward_cl(batch_origin_x, batch1.edge_index, batch_origin_edge, batch1.batch)

        loss_cl1 = model.loss_cl(graph_rep, graph_rep1)
        pred_node1 = linear_pred_atoms1(node_rep1[batch1.masked_atom_indices])
        loss_mask = criterion(pred_node1.double(), labels1)

        acc_node = compute_accuracy(pred_node1, labels1)
        acc_node_accum += acc_node

        if args.mask_edge:
            masked_edge_index1 = batch1.edge_index[:, batch1.connected_edge_indices]
            edge_rep1 = node_rep1[masked_edge_index1[0]] + node_rep1[masked_edge_index1[1]]
            pred_edge1= linear_pred_bonds1(edge_rep1)
            loss_mask += criterion(pred_edge1.double(), batch1.mask_edge_label[:,0])

            acc_edge1 = compute_accuracy(pred_edge1, batch1.mask_edge_label[:,0])
            acc_edge = acc_edge1
            acc_edge_accum += acc_edge
        
        graph_grad=torch.autograd.grad(loss_cl1, model.parameters(), create_graph=True)
        graph_rep2 = gen_ran_output(batch_origin_x, batch1.edge_index, batch_origin_edge, batch1.batch, model, device,graph_grad)

        loss_2 = model.loss_cl(graph_rep, graph_rep2)
 
        #todo
        loss =loss_mask + args.alpha * loss_cl1 + args.beta * loss_2
        optimizer_model.zero_grad()
        
        optimizer_linear_pred_atoms1.zero_grad()
        optimizer_linear_pred_bonds1.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms1.step()
        optimizer_linear_pred_bonds1.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f} tacc: {acc_node:.4f}")

    return loss_accum/step, acc_node_accum/step, acc_edge_accum/step

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--num_tokens', type=int, default=512,
                        help='number of atom tokens (default: 512)') 
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--edge", type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=1,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--output_model_file', type=str, default = './model_gin/', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--alpha', type=float, default = 1, help='loss_maskgcl = alpha * loss_mask + (1 - alpha) * loss_cl1')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset) 
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model = graphcl(gnn).to(device)
    if not args.input_model_file == "":
        model.gnn.from_pretrained(args.input_model_file)

    linear_pred_atoms1 = torch.nn.Linear(args.emb_dim, 512).to(device)
    linear_pred_bonds1 = torch.nn.Linear(args.emb_dim, 4).to(device)
    model_list = [model, linear_pred_atoms1, linear_pred_bonds1]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms1 = optim.Adam(linear_pred_atoms1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds1 = optim.Adam(linear_pred_bonds1.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms1, optimizer_linear_pred_bonds1]
    train_acc_list = []
    train_loss_list = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_bond = train(args, epoch, model_list, dataset, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc_atom)    
    df = pd.DataFrame({'train_acc':train_acc_list,'train_loss':train_loss_list})
    df.to_csv('./logs/logs.csv')   

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + f".pth")

if __name__ == "__main__":
    main()
