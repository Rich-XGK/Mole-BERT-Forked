from random import shuffle
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderMasking, DataLoaderMaskingPred  # , DataListLoader
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
import timeit

triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)
criterion = nn.CrossEntropyLoss()


class GraphConLearning(nn.Module):
    def __init__(self, gnn):
        super(GraphConLearning, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, graph_representation, edge_index, edge_attr, batch):
        node_representation = self.gnn(graph_representation, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        graph_representation = self.projection_head(graph_representation)
        return node_representation, graph_representation

    def loss_con(self, x1, x2):
        """Computes commonly-used contrastive loss L_con"""

        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum("i,j->ij", x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        return loss

    def loss_tri(self, x, x1, x2):
        """Computes commonly-used triplet loss L_tri"""
        loss = triplet_loss(x, x1, x2)
        return loss


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x):
        encoding_indices = self.get_code_indices(x)
        print(encoding_indices[:5])
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight**2, dim=1) - 2.0 * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def train(args, epoch, model_list, tokenizer, dataset, optimizer_list, device):
    model, linear_pred_atoms1, linear_pred_bonds1, linear_pred_atoms2, linear_pred_bonds2 = model_list
    optimizer_model, optimizer_linear_pred_atoms1, optimizer_linear_pred_bonds1, optimizer_linear_pred_atoms2, optimizer_linear_pred_bonds2 = optimizer_list
    model.train()
    linear_pred_atoms1.train()
    linear_pred_bonds1.train()
    linear_pred_atoms2.train()
    linear_pred_bonds2.train()

    loss_total = 0
    acc_node_total = 0
    acc_edge_total = 0

    dataset1 = dataset.shuffle()
    dataset2 = copy.deepcopy(dataset1)
    # get different loader that has different masking rate
    loader1 = DataLoaderMaskingPred(
        dataset1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, mask_rate=args.mask_rate1, mask_edge=args.mask_edge
    )
    loader2 = DataLoaderMaskingPred(
        dataset2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, mask_rate=args.mask_rate2, mask_edge=args.mask_edge
    )
    epoch_iter = tqdm(zip(loader1, loader2), desc="Iteration")

    for step, batch in enumerate(epoch_iter):
        # these two batches are from the same dataset have same sample but different masking rate.
        batch1, batch2 = batch
        masked_edge_label, masked_node_label, masked_atom_indices = batch1["mask_edge_label"], batch1["mask_node_label"], batch1["masked_atom_indices"]
        # batch1 : BatchMasking
        # mask_edge_label: [num_masked_edges, batch_size]
        # mask_node_label: [num_masked_nodes, batch_size]
        # masked_atom_indices: [num_masked_nodes]   NOTE: why? because pyg batch graphs to a big graph.
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        # get node and graph representation from different masking rate.
        node_rep1, graph_rep1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        node_rep2, graph_rep2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        # node_rep: [num_nodes, hidden_size]
        # graph_rep: [num_graphs, hidden_size]
        # NOTE: get commonly-used contrastive loss L_con
        loss_con = model.loss_con(graph_rep1, graph_rep2)

        # get atom ids (label) for masked nodes and compute original graph representation of the graph without masking.
        with torch.no_grad():
            batch_origin_x = copy.deepcopy(batch1.x)  # [num_nodes, num_node_features] = [num_nodes, 2]
            batch_origin_x[batch1.masked_atom_indices] = batch1.mask_node_label  # recover the original node features
            batch_origin_edge = copy.deepcopy(batch1.edge_attr)  # [num_edges, num_edge_features] = [num_edges, 2]
            batch_origin_edge[batch1.connected_edge_indices] = batch1.mask_edge_label  # recover the original edge features
            batch_origin_edge[batch1.connected_edge_indices + 1] = batch1.mask_edge_label  # NOTE: why + 1?
            atom_ids = tokenizer["quantizer"].get_code_indices(tokenizer["encoder"](batch_origin_x, batch1.edge_index, batch_origin_edge))
            # atom_ids: [num_nodes]
            labels1 = atom_ids[batch1.masked_atom_indices]  # [num_masked_nodes], get the atom id (label) of masked nodes
            labels2 = atom_ids[batch2.masked_atom_indices]  # [num_masked_nodes], get the atom id (label) of masked nodes
            # get original graph representation
            _, graph_rep = model.forward_cl(batch_origin_x, batch1.edge_index, batch_origin_edge, batch1.batch)

        # NOTE: get triplet loss L_tri
        loss_tri = model.loss_tri(graph_rep, graph_rep1, graph_rep2)
        # graph_rep: original graph representation without masking
        # graph_rep1: graph representation of masked nodes with masking rate 1
        # graph_rep2: graph representation of masked nodes with masking rate 2

        # NOTE: get TMCL loss L_tmcl = L_con + μ * L_tri (μ = 0.1 in the paper)
        loss_tmcl = loss_con + 0.1 * loss_tri

        # NOTE: get MAM loss for different masking rate.
        # get predicted atom id (label) of masked nodes with masking rate 1
        pred_node1 = linear_pred_atoms1(node_rep1[batch1.masked_atom_indices])
        loss_mam = criterion(pred_node1.double(), labels1)
        # get predicted atom id (label) of masked nodes with masking rate 2
        pred_node2 = linear_pred_atoms2(node_rep2[batch2.masked_atom_indices])
        loss_mam += criterion(pred_node2.double(), labels2)

        # compute accuracy of masked nodes
        acc_node1 = compute_accuracy(pred_node1, labels1)
        acc_node2 = compute_accuracy(pred_node2, labels2)
        acc_node = (acc_node1 + acc_node2) * 0.5
        acc_node_total += acc_node

        if args.mask_edge:
            masked_edge_index1 = batch1.edge_index[:, batch1.connected_edge_indices]
            edge_rep1 = node_rep1[masked_edge_index1[0]] + node_rep1[masked_edge_index1[1]]
            pred_edge1 = linear_pred_bonds1(edge_rep1)
            loss_mam += criterion(pred_edge1.double(), batch1.mask_edge_label[:, 0])

            masked_edge_index2 = batch2.edge_index[:, batch2.connected_edge_indices]
            edge_rep2 = node_rep2[masked_edge_index2[0]] + node_rep2[masked_edge_index2[1]]
            pred_edge2 = linear_pred_bonds2(edge_rep2)
            loss_mam += criterion(pred_edge2.double(), batch2.mask_edge_label[:, 0])

            acc_edge1 = compute_accuracy(pred_edge1, batch1.mask_edge_label[:, 0])
            acc_edge2 = compute_accuracy(pred_edge2, batch2.mask_edge_label[:, 0])
            acc_edge = (acc_edge1 + acc_edge2) * 0.5
            acc_edge_total += acc_edge

        # NOTE: get total MoleBERT pre-training loss L_molebert = L_tmcl + L_mam
        loss_molebert = loss_tmcl + loss_mam
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms1.zero_grad()
        optimizer_linear_pred_bonds1.zero_grad()
        optimizer_linear_pred_atoms2.zero_grad()
        optimizer_linear_pred_bonds2.zero_grad()

        loss_molebert.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms1.step()
        optimizer_linear_pred_bonds1.step()
        optimizer_linear_pred_atoms2.step()
        optimizer_linear_pred_bonds2.step()

        loss_total += float(loss_molebert.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} total_loss: {loss_molebert:.4f} total_acc: {acc_node:.4f}")

    return loss_total / step, acc_node_total / step, acc_edge_total / step


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of pre-training of graph neural networks")
    parser.add_argument("--device", type=int, default=0, help="which gpu to use if any (default: 0)")
    parser.add_argument("--batch_size", type=int, default=2, help="input batch size for training (default: 256)")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--decay", type=float, default=0, help="weight decay (default: 0)")
    parser.add_argument("--num_layer", type=int, default=5, help="number of GNN message passing layers (default: 5).")
    parser.add_argument("--emb_dim", type=int, default=300, help="embedding dimensions (default: 300)")
    parser.add_argument("--num_tokens", type=int, default=512, help="number of atom tokens (default: 512)")
    parser.add_argument("--edge", type=int, default=1)
    parser.add_argument("--dropout_ratio", type=float, default=0, help="dropout ratio (default: 0)")
    parser.add_argument("--mask_rate1", type=float, default=0.15, help="dropout ratio (default: 0.15)")
    parser.add_argument("--mask_rate2", type=float, default=0.30, help="dropout ratio (default: 0.30)")
    parser.add_argument("--mask_edge", type=int, default=1, help="whether to mask edges or not together with atoms")
    parser.add_argument("--JK", type=str, default="last", help="how the node features are combined across layers. last, sum, max or concat")
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent", help="root directory of dataset for pretraining")
    parser.add_argument("--input_model_file", type=str, default="", help="filename to output the model")
    parser.add_argument("--output_model_file", type=str, default="./model_gin/", help="filename to output the model")
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--seed", type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataset loading")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GraphConLearning(gnn).to(device)
    tokenizer_encoder = GNN(args.num_layer, args.emb_dim, gnn_type=args.gnn_type).to(device)
    tokenizer_encoder.from_pretrained(f"./checkpoints/MoleBERT-Tokenizervqencoder.pth")
    tokenizer_quantizer = VectorQuantizer(args.emb_dim, args.num_tokens, commitment_cost=0.25).to(device)
    tokenizer_quantizer.from_pretrained(f"./checkpoints/MoleBERT-Tokenizervqquantizer.pth")
    tokenizer = {"encoder": tokenizer_encoder, "quantizer": tokenizer_quantizer}
    if not args.input_model_file == "":
        model.gnn.from_pretrained(args.input_model_file)

    # atom and bond prediction heads for 15% and 30% masking
    linear_pred_atoms1 = torch.nn.Linear(args.emb_dim, 512).to(device)
    linear_pred_bonds1 = torch.nn.Linear(args.emb_dim, 4).to(device)
    linear_pred_atoms2 = torch.nn.Linear(args.emb_dim, 512).to(device)
    linear_pred_bonds2 = torch.nn.Linear(args.emb_dim, 4).to(device)
    model_list = [model, linear_pred_atoms1, linear_pred_bonds1, linear_pred_atoms2, linear_pred_bonds2]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms1 = optim.Adam(linear_pred_atoms1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds1 = optim.Adam(linear_pred_bonds1.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms2 = optim.Adam(linear_pred_atoms2.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds2 = optim.Adam(linear_pred_bonds2.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms1, optimizer_linear_pred_bonds1, optimizer_linear_pred_atoms2, optimizer_linear_pred_bonds2]
    train_acc_list = []
    train_loss_list = []

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_bond = train(args, epoch, model_list, tokenizer, dataset, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc_atom)
    df = pd.DataFrame({"train_acc": train_acc_list, "train_loss": train_loss_list})
    df.to_csv("./logs/logs.csv")

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + f"Mole-BERT.pth")


if __name__ == "__main__":
    main()
