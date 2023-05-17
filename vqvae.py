"""
    Mole-BERT Tokenizer training.
    A variant of VQ-VAE as a context-aware tokenizer to encode atom attributes into chemically meaningful discrete codes.
    It can enlarge the atom vocabulary size and mitigate the quantitative divergence between dominant (e.g., carbons) and rare atoms (e.g., phosphorus).
"""
import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from model import GNN, GNNDecoder
from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

NUM_NODE_ATTR = 119
NUM_NODE_CHIRAL = 4
NUM_BOND_ATTR = 4

criterion = nn.CrossEntropyLoss()


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

    This module keeps track of a hidden exponential moving average that is
    initialized as a vector of zeros which is then normalized to give the average.
    This gives us a moving average which isn't biased towards either zero or the
    initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

    Initially:
        hidden_0 = 0
    Then iteratively:
        hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
        average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay**self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def forward(self, x):
        encoding_indices = self.get_code_indices(x)  # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (updated_ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            dw = torch.matmul(encodings.t(), x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1)
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()
        print("commitment loss:", loss)
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True) + torch.sum(self.embeddings**2, dim=1) - 2.0 * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)


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

    def forward(self, x, node_rep):
        encoding_indices = self.get_code_indices(x, node_rep)  # x: B * H, encoding_indices: B
        quantized = self.quantize(encoding_indices)
        # VQ loss: move the embeddings towards the encoder's output, update the codebook
        q_latent_loss = F.mse_loss(quantized, node_rep.detach())
        # commitment loss: encourages the output of the encoder to stay close to the chosen codebook embedding, update the encoder
        e_latent_loss = F.mse_loss(node_rep, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = node_rep + (quantized - node_rep).detach().contiguous()
        return quantized, loss

    def get_code_indices(self, x, node_rep):
        # x: node features of original graph with shape (b, 2) and the fist column denotes the atom type.
        # node_rep: node representation of the encoded graph with shape (b, d_embedding).

        # get the indices of C, N, O and others, all bool tensor with shape (b, ).
        atom_type = x[:, 0]
        index_c = atom_type == 5
        index_n = atom_type == 6
        index_o = atom_type == 7
        # in python: ~ denotes not, | denotes or, & denotes and.
        index_others = ~(index_c + index_n + index_o)  # means not one or more of C, N, O.

        # compute L2 distance
        # TODO: what this computation means? How to compute the L2 distance using below code?
        encoding_indices = torch.ones(node_rep.size(0)).long().to(node_rep.device)
        # C: the context-aware C atom embeddings are the first 377 embeddings in the codebook
        distances = (
            torch.sum(node_rep[index_c] ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight[0:377] ** 2, dim=1)
            - 2.0 * torch.matmul(node_rep[index_c], self.embeddings.weight[0:377].t())
        )
        encoding_indices[index_c] = torch.argmin(distances, dim=1)
        # N: the context-aware N atom embeddings are the next 55 embeddings in the codebook
        distances = (
            torch.sum(node_rep[index_n] ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight[378:433] ** 2, dim=1)
            - 2.0 * torch.matmul(node_rep[index_n], self.embeddings.weight[378:433].t())
        )
        encoding_indices[index_n] = torch.argmin(distances, dim=1) + 378
        # O: the context-aware O atom embeddings are the next 54 embeddings in the codebook
        distances = (
            torch.sum(node_rep[index_o] ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight[434:488] ** 2, dim=1)
            - 2.0 * torch.matmul(node_rep[index_o], self.embeddings.weight[434:488].t())
        )
        encoding_indices[index_o] = torch.argmin(distances, dim=1) + 434

        # Others: others: the context-aware other atom embeddings are the last 22 embeddings in the codebook
        distances = (
            torch.sum(node_rep[index_others] ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight[489:511] ** 2, dim=1)
            - 2.0 * torch.matmul(node_rep[index_others], self.embeddings.weight[489:511].t())
        )
        encoding_indices[index_others] = torch.argmin(distances, dim=1) + 489
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file))


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def train_vae(args, epoch, model_list, loader, optimizer_list, device):
    # vq-vae training function

    criterion = nn.CrossEntropyLoss()

    model, vq_layer, dec_pred_atoms, dec_pred_bonds, dec_pred_atoms_chiral = model_list
    optimizer_model, optimizer_model_vq, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_atoms_chiral = optimizer_list

    model.train()
    vq_layer.train()
    dec_pred_atoms.train()
    dec_pred_atoms_chiral.train()

    if dec_pred_bonds is not None:
        dec_pred_bonds.train()

    loss_accumulation = 0
    epoch_iter = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        # get node representation
        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        # get quantized node representation and VQ loss + commitment loss
        # TODO: fix this line --> e, e_q_loss = vq_layer(node_rep, ,node_rep)
        e, e_q_loss = vq_layer(batch.x, node_rep)  # fixed
        pred_node = dec_pred_atoms(e, batch.edge_index, batch.edge_attr)
        pred_node_chiral = dec_pred_atoms_chiral(e, batch.edge_index, batch.edge_attr)

        # compute loss of reconstruction: atom type + chiral type + bond type (if applicable)
        atom_loss = criterion(pred_node, batch.x[:, 0])
        atom_chiral_loss = criterion(pred_node_chiral, batch.x[:, 1])
        recon_loss = atom_loss + atom_chiral_loss

        if args.edge:
            edge_rep = e[batch.edge_index[0]] + e[batch.edge_index[1]]
            pred_edge = dec_pred_bonds(edge_rep, batch.edge_index, batch.edge_attr)
            recon_loss += criterion(pred_edge, batch.edge_attr[:, 0])

        # compute total loss: reconstruction loss + quantization loss ()
        loss = recon_loss + e_q_loss

        # optimize
        optimizer_model.zero_grad()
        optimizer_model_vq.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        optimizer_dec_pred_atoms_chiral.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()
        optimizer_model.step()
        optimizer_model_vq.step()
        optimizer_dec_pred_atoms.step()
        optimizer_dec_pred_atoms_chiral.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accumulation += float(loss.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} train_loss: {loss.item():.4f}")

    return loss_accumulation / step


def main():
    parser = argparse.ArgumentParser(description="PyTorch implementation of pre-training of graph neural networks")
    parser.add_argument("--device", type=int, default=5, help="which gpu to use if any (default: 0)")
    parser.add_argument("--batch_size", type=int, default=256, help="input batch size for training (default: 256)")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--decay", type=float, default=0, help="weight decay (default: 0)")
    parser.add_argument("--num_layer", type=int, default=5, help="number of GNN message passing layers (default: 5).")

    parser.add_argument("--emb_dim", type=int, default=300, help="embedding dimensions (default: 300)")
    parser.add_argument("--num_tokens", type=int, default=512, help="number of atom tokens (default: 512)")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="commitment_cost")  # what is this?
    parser.add_argument("--edge", type=int, default=1, help="whether to decode edges or not together with atoms")

    parser.add_argument("--dropout_ratio", type=float, default=0.0, help="dropout ratio (default: 0)")
    parser.add_argument("--mask_rate", type=float, default=0.0, help="dropout ratio (default: 0.15)")
    parser.add_argument("--JK", type=str, default="last", help="how the node features are combined across layers. last, sum, max or concat")
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent", help="root directory of dataset for pre-training")
    parser.add_argument("--output_model_file", type=str, default="", help="filename to output the model")
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--seed", type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataset loading")
    parser.add_argument("--input_model_file", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    args = parser.parse_args()
    print(args)

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.edge))

    # set up dataset and transform function.
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    # molecule example: Data(x=[30, 2], id=[1], edge_index=[2, 66], edge_attr=[66, 2])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ###
    # set up components for vq-vae and training.

    # Encoder: encode node features to get context-aware node representations / atom embeddings
    model = GNN(args.num_layer, args.emb_dim).to(device)
    if args.input_model_file is not None and args.input_model_file != "":
        model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    # Quantizer: looks up the nearest code in codebook for each encoder output(node representation / atom embedding)
    # to get discrete latent representation (node's index in codebook / dictionary) and the corresponding embedding vector
    vq_layer = VectorQuantizer(args.emb_dim, args.num_tokens, args.commitment_cost).to(device)

    # Decoder: reconstruct the input molecule graph.
    # reconstruct atom types
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    # reconstruct atom chirality
    atom_chiral_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_CHIRAL, JK=args.JK, gnn_type=args.gnn_type).to(device)
    if args.edge:  # whether to decode edges or not together with atoms
        # reconstruct bond types
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type="linear").to(device)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [model, vq_layer, atom_pred_decoder, bond_pred_decoder, atom_chiral_pred_decoder]
    ###

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_model_vq = optim.Adam(vq_layer.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms_chiral = optim.Adam(atom_chiral_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_model_vq, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_atoms_chiral]

    # set up scheduler
    if args.use_scheduler:
        print("--------- Use scheduler -----------")

        def scheduler(epoch):
            return (1 + np.cos((epoch) * np.pi / args.epochs)) * 0.5

        scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
        scheduler_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_dec_chiral = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms_chiral, lr_lambda=scheduler)
        scheduler_list = [scheduler_model, scheduler_dec, scheduler_dec_chiral, None]
    else:
        scheduler_model = None
        scheduler_dec = None

    output_file_temp = "./checkpoints/" + args.output_model_file
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train_loss = train_vae(args, epoch, model_list, loader, optimizer_list, device)
        if not resume:
            # if epoch == 30:
            if epoch == 1:
                torch.save(model.state_dict(), output_file_temp + f"vqencoder.pth")
                torch.save(vq_layer.state_dict(), output_file_temp + f"vqquantizer.pth")
        print(train_loss)
        if scheduler_model is not None:
            scheduler_model.step()
        if scheduler_dec is not None:
            scheduler_dec.step()

    output_file = "./checkpoints/" + args.output_model_file
    os.mkdir(output_file) if not os.path.exists(output_file) else None

    if resume:
        torch.save(model.state_dict(), args.input_model_file.rsplit(".", 1)[0] + f"_resume_{args.epochs}_{args.start_epoch}.pth")
    elif not args.output_model_file == "":
        torch.save(model.state_dict(), output_file + ".pth")


if __name__ == "__main__":
    main()
