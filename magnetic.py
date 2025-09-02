import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl import AddSelfLoop
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, auc, roc_curve, balanced_accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter
import time
import pandas as pd
import os
import statistics
import warnings
from models import AGDNConv

warnings.filterwarnings("ignore")
th.autograd.set_detect_anomaly(True)


def set_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    random.seed(seed)


def from_edge_list_to_matrix(u_list, v_list, mat_shape):
    inter_matrix = np.zeros(mat_shape)
    for u, v in zip(u_list, v_list):
        inter_matrix[u, v] = 1
    return inter_matrix


def from_matrix_to_edge_list(mat, mode='int', value=1, include_zeros=True):
    # Convert the adjacency matrix to edge list
    u = []
    v = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mode == 'int' and mat[i, j] == value:
                u.append(i)
                v.append(j)
            elif mode == 'float':
                if include_zeros:
                    u.append(i)
                    v.append(j)
                else:
                    if mat[i][j] != 0:
                        u.append(i)
                        v.append(j)
    return np.array(u), np.array(v)


def from_edge_list_to_tuple_list(u, v):
    drug_target_int, drug_target_reverse = [], []
    for u_el, v_el in zip(u, v):
        drug_target_int.append((u_el, v_el))
        drug_target_reverse.append((v_el, u_el))
    return drug_target_int, drug_target_reverse


def train_val_test_split(u, v, seed=42):
    eids = np.arange(len(u))  # Getting an 1-d array that contains all node indexes
    eids = np.random.RandomState(seed=seed).permutation(eids)  # Suffling the node indexes
    test_size = int(
        len(eids) * 0.2)  # Getting the test size (number of testing edges), thus the index of the last testing node
    val_size = int(len(eids) * 0.1)
    train_size = len(u) - test_size - val_size  # Getting the training size (number of training edges)
    # Split edge set for training and testing
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size: test_size + val_size]], v[eids[test_size: test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]
    folds = [{
        'train_set': (train_pos_u, train_pos_v),
        'val_set': (val_pos_u, val_pos_v),
        'test_set': (test_pos_u, test_pos_v)
    }]
    return folds


def extract_s2_int_list(rows, luo_interactions):
    temp_luo_interactions = np.copy(luo_interactions)
    for i in range(temp_luo_interactions.shape[0]):
        if i not in rows:
            for j in range(temp_luo_interactions.shape[1]):
                temp_luo_interactions[i, j] = 0
    pos_u, pos_v = from_matrix_to_edge_list(temp_luo_interactions)

    neg_u, neg_v = [], []
    for i in rows:
        for j in range(temp_luo_interactions.shape[1]):
            if temp_luo_interactions[i, j] == 0:
                neg_u.append(i)
                neg_v.append(j)

    neg_u = np.asarray(neg_u)
    neg_v = np.asarray(neg_v)
    eids = np.arange(len(neg_u))
    eids = np.random.RandomState(seed=seed).permutation(eids)
    neg_u = neg_u[eids]
    neg_v = neg_v[eids]
    return pos_u, pos_v, neg_u, neg_v


def extract_s3_int_list(cols, luo_interactions):
    temp_luo_interactions = np.copy(luo_interactions)

    for j in range(temp_luo_interactions.shape[1]):
        if j not in cols:
            for i in range(temp_luo_interactions.shape[0]):
                temp_luo_interactions[i, j] = 0
    pos_u, pos_v = from_matrix_to_edge_list(temp_luo_interactions)

    neg_u, neg_v = [], []
    for j in cols:
        for i in range(temp_luo_interactions.shape[0]):
            if temp_luo_interactions[i, j] == 0:
                neg_u.append(i)
                neg_v.append(j)

    neg_u = np.asarray(neg_u)
    neg_v = np.asarray(neg_v)
    eids = np.arange(len(neg_u))
    eids = np.random.RandomState(seed=seed).permutation(eids)
    neg_u = neg_u[eids]
    neg_v = neg_v[eids]
    return pos_u, pos_v, neg_u, neg_v


def sample_datapoints(neg_u, neg_v, pos_u, neg_ratio):
    # Sampling on negative data points
    if neg_ratio:
        n_neg_samples = int(len(pos_u) * neg_ratio)
        neg_u, neg_v = neg_u[:n_neg_samples], neg_v[:n_neg_samples]
    return neg_u, neg_v


def process_affinities(train_pos_u, train_pos_v, drug_target_aff_ic50_mat, drug_target_aff_kd_mat,
                       drug_target_aff_ki_mat):
    aff_ic50_mat = np.zeros(drug_target_aff_ic50_mat.shape)
    aff_kd_mat = np.zeros(drug_target_aff_ic50_mat.shape)
    aff_ki_mat = np.zeros(drug_target_aff_ic50_mat.shape)
    for i, j in zip(train_pos_u, train_pos_v):
        if drug_target_aff_ic50_mat[i, j] != 0:
            aff_ic50_mat[i, j] = 1 - drug_target_aff_ic50_mat[i, j]
        if drug_target_aff_kd_mat[i, j] != 0:
            aff_kd_mat[i, j] = 1 - drug_target_aff_kd_mat[i, j]
        if drug_target_aff_ki_mat[i, j] != 0:
            aff_ki_mat[i, j] = 1 - drug_target_aff_ki_mat[i, j]
    return aff_ic50_mat, aff_kd_mat, aff_ki_mat


def cross_val_split(luo_interactions, setting, n_folds, val_ratio=0.1, neg_ratio=10, seed=42):
    if setting == 's1':
        u, v = from_matrix_to_edge_list(luo_interactions)
        neg_u, neg_v = find_all_negative_edges( luo_interactions)  # Find all negative (non existing) edges and split them for training and testing

        eids = np.arange(len(u))
        eids = np.random.RandomState(seed=seed).permutation(eids)
        kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        folds = []
        for train_val_index, test_index in kf.split(eids):
            val_size = int(len(train_val_index) * val_ratio)
            train_val_pos_u, train_val_pos_v = u[eids[train_val_index]], v[eids[train_val_index]]
            val_pos_u, val_pos_v = train_val_pos_u[:val_size], train_val_pos_v[:val_size]
            train_pos_u, train_pos_v = train_val_pos_u[val_size:], train_val_pos_v[val_size:]
            test_pos_u, test_pos_v = u[eids[test_index]], v[eids[test_index]]

            test_neg_u, test_neg_v, val_neg_u, val_neg_v, train_neg_u, train_neg_v = train_val_test_split_neg(n_folds, val_ratio,neg_u,neg_v)
            train_neg_u, train_neg_v = sample_datapoints(train_neg_u, train_neg_v, train_pos_u, neg_ratio)

            fold = {
                'train_set_pos': (train_pos_u, train_pos_v),
                'train_set_neg': (train_neg_u, train_neg_v),
                'val_set_pos': (val_pos_u, val_pos_v),
                'val_set_neg': (val_neg_u, val_neg_v),
                'test_set_pos': (test_pos_u, test_pos_v),
                'test_set_neg': (test_neg_u, test_neg_v),
            }
            folds.append(fold)

    if setting == 's2':

        eids = np.arange(luo_interactions.shape[0])
        eids = np.random.RandomState(seed=seed).permutation(eids)
        kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        folds = []
        for train_val_index, test_index in kf.split(eids):
            val_size = int(len(train_val_index) * val_ratio)
            train_val_rows = eids[train_val_index]
            val_rows = train_val_rows[:val_size]
            train_rows = train_val_rows[val_size:]
            test_rows = eids[test_index]

            train_pos_u, train_pos_v, train_neg_u, train_neg_v = extract_s2_int_list(train_rows, luo_interactions)
            val_pos_u, val_pos_v, val_neg_u, val_neg_v = extract_s2_int_list(val_rows, luo_interactions)
            test_pos_u, test_pos_v, test_neg_u, test_neg_v = extract_s2_int_list(test_rows, luo_interactions)

            train_neg_u, train_neg_v = sample_datapoints(train_neg_u, train_neg_v, train_pos_u, neg_ratio)

            fold = {
                'train_set_pos': (train_pos_u, train_pos_v),
                'train_set_neg': (train_neg_u, train_neg_v),
                'val_set_pos': (val_pos_u, val_pos_v),
                'val_set_neg': (val_neg_u, val_neg_v),
                'test_set_pos': (test_pos_u, test_pos_v),
                'test_set_neg': (test_neg_u, test_neg_v),

            }
            folds.append(fold)

    if setting == 's3':

        eids = np.arange(luo_interactions.shape[1])
        eids = np.random.RandomState(seed=seed).permutation(eids)
        kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
        folds = []
        for train_val_index, test_index in kf.split(eids):
            val_size = int(len(train_val_index) * val_ratio)
            train_val_cols = eids[train_val_index]
            val_cols = train_val_cols[:val_size]
            train_cols = train_val_cols[val_size:]
            test_cols = eids[test_index]

            train_pos_u, train_pos_v, train_neg_u, train_neg_v = extract_s3_int_list(train_cols, luo_interactions)
            val_pos_u, val_pos_v, val_neg_u, val_neg_v = extract_s3_int_list(val_cols, luo_interactions)
            test_pos_u, test_pos_v, test_neg_u, test_neg_v = extract_s3_int_list(test_cols, luo_interactions)

            train_neg_u, train_neg_v = sample_datapoints(train_neg_u, train_neg_v, train_pos_u, neg_ratio)

            fold = {
                'train_set_pos': (train_pos_u, train_pos_v),
                'train_set_neg': (train_neg_u, train_neg_v),
                'val_set_pos': (val_pos_u, val_pos_v),
                'val_set_neg': (val_neg_u, val_neg_v),
                'test_set_pos': (test_pos_u, test_pos_v),
                'test_set_neg': (test_neg_u, test_neg_v),

            }
            folds.append(fold)

    if setting == 's4':

        folds = []
        for fold in range(n_folds):
            d_eids = np.arange(luo_interactions.shape[0])
            d_eids = np.random.RandomState(seed=fold).permutation(d_eids)
            t_eids = np.arange(luo_interactions.shape[1])
            t_eids = np.random.RandomState(seed=fold).permutation(t_eids)

            n_train_val_d = int(0.7 * luo_interactions.shape[0])
            n_train_d = int(0.9 * n_train_val_d)
            train_d = d_eids[:n_train_d]
            val_d = d_eids[n_train_d:n_train_val_d]
            test_d = d_eids[n_train_val_d:]

            n_train_val_t = int(0.7 * luo_interactions.shape[1])
            n_train_t = int(0.9 * n_train_val_t)
            train_t = t_eids[:n_train_t]
            val_t = t_eids[n_train_t: n_train_t + n_train_val_t - n_train_t]
            test_t = t_eids[n_train_val_t:]

            train_pos_u, train_pos_v = [], []
            train_neg_u, train_neg_v = [], []
            for t_d in train_d:
                for t_t in train_t:
                    if luo_interactions[t_d, t_t] == 1:
                        train_pos_u.append(t_d)
                        train_pos_v.append(t_t)
                    else:
                        train_neg_u.append(t_d)
                        train_neg_v.append(t_t)

            val_pos_u, val_pos_v = [], []
            val_neg_u, val_neg_v = [], []
            for t_d in val_d:
                for t_t in val_t:
                    if luo_interactions[t_d, t_t] == 1:
                        val_pos_u.append(t_d)
                        val_pos_v.append(t_t)
                    else:
                        val_neg_u.append(t_d)
                        val_neg_v.append(t_t)

            test_pos_u, test_pos_v = [], []
            test_neg_u, test_neg_v = [], []
            for t_d in test_d:
                for t_t in test_t:
                    if luo_interactions[t_d, t_t] == 1:
                        test_pos_u.append(t_d)
                        test_pos_v.append(t_t)
                    else:
                        test_neg_u.append(t_d)
                        test_neg_v.append(t_t)

            fold = {
                'train_set_pos': (train_pos_u, train_pos_v),
                'train_set_neg': (train_neg_u, train_neg_v),
                'val_set_pos': (val_pos_u, val_pos_v),
                'val_set_neg': (val_neg_u, val_neg_v),
                'test_set_pos': (test_pos_u, test_pos_v),
                'test_set_neg': (test_neg_u, test_neg_v),
            }
            folds.append(fold)

    return folds


def find_all_negative_edges(luo_interactions):
    u = []
    v = []
    for i in range(luo_interactions.shape[0]):
        for j in range(luo_interactions.shape[1]):
            if luo_interactions[i, j] == 0:  # or i == j)
                u.append(i)
                v.append(j)
    return np.array(u), np.array(v)


def train_val_test_split_neg(n_folds, val_ratio, neg_u, neg_v):
    if n_folds == 1:
        test_ratio = 0.2
        val_ratio = 0.1
    else:
        test_ratio = 1 / n_folds

    test_size = int(
        len(neg_u) * test_ratio)  # Getting the test size (number of testing edges), thus the index of the last testing node
    val_size = int((len(neg_u) - test_size) * val_ratio)
    neg_eids = np.arange(len(neg_u))
    neg_eids = np.random.RandomState(seed=seed).permutation(neg_eids)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size: test_size + val_size]], neg_v[
        neg_eids[test_size: test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]
    return test_neg_u, test_neg_v, val_neg_u, val_neg_v, train_neg_u, train_neg_v


def set_edge_features(pos_edge_graph, edge_info, affinity_layer_integer):
    # Edge features
    edge_features = {}
    if 'is_similar_drug' in edge_info:
        edge_weights = th.tensor(edge_info['is_similar_drug'].flatten()).to(th.float32)
        pos_edge_graph.edges['is_similar_drug'].data['weight'] = edge_weights[
            edge_weights.nonzero()] if affinity_layer_integer else edge_weights
        edge_features['is_similar_drug'] = {'edge_weight': pos_edge_graph.edges['is_similar_drug'].data['weight']}
    if 'is_similar_target' in edge_info:
        edge_weights = th.tensor(edge_info['is_similar_target'].flatten()).to(th.float32)
        pos_edge_graph.edges['is_similar_target'].data['weight'] = edge_weights[
            edge_weights.nonzero()] if affinity_layer_integer else edge_weights
        edge_features['is_similar_target'] = {'edge_weight': pos_edge_graph.edges['is_similar_target'].data['weight']}
    if 'drug_target_aff_ic50_mat' in edge_info:
        edge_weights = th.tensor(edge_info['drug_target_aff_ic50_mat'].flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_IC50'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights
        edge_weights = th.tensor(edge_info['drug_target_aff_ic50_mat'].T.flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_IC50_reverse'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights

        edge_features['interacts_IC50'] = {'edge_weight': pos_edge_graph.edges['interacts_IC50'].data['weight']}
        edge_features['interacts_IC50_reverse'] = {
            'edge_weight': pos_edge_graph.edges['interacts_IC50_reverse'].data['weight']}
    if 'drug_target_aff_kd_mat' in edge_info:
        edge_weights = th.tensor(edge_info['drug_target_aff_kd_mat'].flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_Kd'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights
        edge_weights = th.tensor(edge_info['drug_target_aff_kd_mat'].T.flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_Kd_reverse'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights

        edge_features['interacts_Kd'] = {'edge_weight': pos_edge_graph.edges['interacts_Kd'].data['weight']}
        edge_features['interacts_Kd_reverse'] = {
            'edge_weight': pos_edge_graph.edges['interacts_Kd_reverse'].data['weight']}
    if 'drug_target_aff_ki_mat' in edge_info:
        edge_weights = th.tensor(edge_info['drug_target_aff_ki_mat'].flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_Ki'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights
        edge_weights = th.tensor(edge_info['drug_target_aff_ki_mat'].T.flatten()).to(th.float32)
        pos_edge_graph.edges['interacts_Ki_reverse'].data['weight'] = edge_weights[
            edge_weights.nonzero()]  # if affinity_layer_integer else edge_weights

        edge_features['interacts_Ki'] = {'edge_weight': pos_edge_graph.edges['interacts_Ki'].data['weight']}
        edge_features['interacts_Ki_reverse'] = {
            'edge_weight': pos_edge_graph.edges['interacts_Ki_reverse'].data['weight']}

    for i in range(2, 5):
        relation = 'is_similar_drug_' + str(i)
        if relation in edge_info:
            edge_weights = th.tensor(edge_info[relation].flatten()).to(th.float32)
            pos_edge_graph.edges[relation].data['weight'] = edge_weights
            edge_features[relation] = {'edge_weight': edge_weights}

    for i in range(2, 4):
        relation = 'is_similar_target_' + str(i)
        if relation in edge_info:
            edge_weights = th.tensor(edge_info[relation].flatten()).to(th.float32)
            pos_edge_graph.edges[relation].data['weight'] = edge_weights
            edge_features[relation] = {'edge_weight': edge_weights}

    return edge_features


def set_node_features(pos_edge_graph, feat_size, feat_type):
    # Node features
    n_drugs = pos_edge_graph.num_nodes('drug')
    n_targets = pos_edge_graph.num_nodes('target')

    # Drug features
    if feat_type['drug'] == 'loaded_luo':
        drug_features = pd.read_csv('features/drug_vector_d100.txt', sep='\t', header=None).to_numpy(dtype=np.double)
        pos_edge_graph.nodes['drug'].data['h'] = th.tensor(drug_features).float()
    elif feat_type['drug'] == 'loaded_emb':
        drug_features = pd.read_csv('features/david_drug.txt', sep='\t', header=None).to_numpy(dtype=np.double)
        pos_edge_graph.nodes['drug'].data['h'] = th.tensor(drug_features).float()
    elif feat_type['drug'] == 'ecfp4':
        drug_features = pd.read_csv('features/ecfp4_1024_luo.txt', sep=',', header=None).to_numpy(dtype=np.double)
        pos_edge_graph.nodes['drug'].data['h'] = th.tensor(drug_features).float()
    elif feat_type['drug'] == 'random':
        pos_edge_graph.nodes['drug'].data['h'] = th.randn(n_drugs, feat_size)
    elif feat_type['drug'] == '1hot':
        pos_edge_graph.nodes['drug'].data['h'] = th.tensor(np.eye(n_drugs)).to(th.float32)

    # Target features
    if feat_type['target'] == 'loaded_luo':
        target_features = pd.read_csv('features/protein_vector_d400.txt', sep='\t', header=None).to_numpy(
            dtype=np.double)
        pos_edge_graph.nodes['target'].data['h'] = th.tensor(target_features).float()
    elif feat_type['target'] == 'loaded_emb':
        target_features = pd.read_csv('features/protein_embs_bert_d1024.csv', sep='\t', header=None).to_numpy(
            dtype=np.double)  # embs.txt
        pos_edge_graph.nodes['target'].data['h'] = th.tensor(target_features).float()
    elif feat_type['target'] == 'loaded_luo+loaded_emb':
        target_features_emb = pd.read_csv('features/embs.txt', sep='\t', header=None).to_numpy(dtype=np.double)
        target_features_luo = pd.read_csv('features/protein_vector_d400.txt', sep='\t', header=None).to_numpy(
            dtype=np.double)
        target_features = np.concatenate([target_features_emb, target_features_luo], axis=1)
        pos_edge_graph.nodes['target'].data['h'] = th.tensor(target_features).float()
    elif feat_type['target'] == 'random':
        pos_edge_graph.nodes['target'].data['h'] = th.randn(n_targets, feat_size)
    elif feat_type['target'] == '1hot':
        pos_edge_graph.nodes['target'].data['h'] = th.tensor(np.eye(n_targets)).to(th.float32)

    drug_dim = pos_edge_graph.nodes['drug'].data['h'].shape[1]
    target_dim = pos_edge_graph.nodes['target'].data['h'].shape[1]

    vec_size = max([drug_dim, target_dim])

    if drug_dim < vec_size:
        pos_edge_graph.nodes['drug'].data['h'] = th.cat(
            (pos_edge_graph.nodes['drug'].data['h'], th.zeros(n_drugs, vec_size - drug_dim)), 1).to(th.float32)
    if target_dim < vec_size:
        pos_edge_graph.nodes['target'].data['h'] = th.cat(
            (pos_edge_graph.nodes['target'].data['h'], th.zeros(n_targets, vec_size - target_dim)), 1).to(th.float32)



def construct_negative_graph(graph, etype, neg_src=None, neg_dst=None):
    utype, relation, vtype = etype

    return dgl.heterograph({etype: (neg_src, neg_dst), (vtype, relation, utype): (neg_dst, neg_src)},
                           num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


def pred2(logits, inter, neg_inter):
    h = th.mm(logits['drug'], logits['target'].T)

    train_pos_scores = th.mul(h, inter).flatten()
    train_pos_scores = train_pos_scores[th.nonzero(train_pos_scores).squeeze()]
    train_neg_scores = th.mul(h, neg_inter).flatten()
    train_pos_scores = train_pos_scores[th.nonzero(train_pos_scores).squeeze()]
    return train_pos_scores, train_neg_scores


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            # h[etype[0]] = F.dropout(h[etype[0]], 0.1)
            # h[etype[2]] = F.dropout(h[etype[2]], 0.1)
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            return graph.edges[etype].data['score']


class HeteroDistMultPredictor(nn.Module):
    def forward(self, graph, h, etype):
        pass


class HeteroDotProductVariationalPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

            return graph.edges[etype].data['score']


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.W1 = nn.Linear(in_dims * 2, out_dims)


    def apply_edges(self, edges):
        x = th.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W1(x)
        return {'score': y}

    def forward(self, graph, h, etype):

        with graph.local_scope():
            graph.ndata['h'] = h  # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, gnn_layer, n_layers, model_drug, model_target):
        super().__init__()
        self.in_feats, self.hid_feats, self.out_feats = in_feats, hid_feats, out_feats
        self.architecture = gnn_layer['architecture']
        self.gnn_layer = gnn_layer
        self.layers = []
        self.do = nn.Dropout(0.1)
        self.model_drug, self.model_target = model_drug, model_target
        self.drug_weigts = None
        self.target_weigts = None
        self.div_reg = 0


        self.att_linear_drugs3 = nn.Linear(out_feats * 5, out_feats)
        self.att_linear_targets3 = nn.Linear(out_feats * 4, out_feats)
        self.att_linear_drugs4 = nn.Linear(hid_feats * 5, hid_feats)
        self.att_linear_targets4 = nn.Linear(hid_feats * 4, hid_feats)

        self.first_layer = dglnn.HeteroGraphConv({
            rel: self.get_layers(gnn_layer, n_layers, in_feats, hid_feats, out_feats)[0]
            for rel in rel_names}, aggregate=self.my_agg_func2)  # 'sum', self.my_agg_func2
        self.layers.append(self.first_layer)

        for i in range(n_layers - 2):
            self.layer = dglnn.HeteroGraphConv({
                rel: self.get_layers(gnn_layer, n_layers, in_feats, hid_feats, out_feats)[i + 1]
                for rel in rel_names}, aggregate=self.my_agg_func2)
            self.layers.append(self.layer)

        self.last_layer = dglnn.HeteroGraphConv({
            rel: self.get_layers(gnn_layer, n_layers, in_feats, hid_feats, out_feats)[-1]
            for rel in rel_names}, aggregate=self.my_agg_func2)
        # self.layers.append(self.last_layer)

    def get_layers(self, gnn_layer, n_layers, in_feats, hid_feats, out_feats):

        layers = [AGDNConv(in_feats, hid_feats, gnn_layer['n_heads'], 3, residual=True, attn_drop=0, pos_emb=False,
                           weight_style="HA", hop_norm=True), ]
        return layers

    def forward(self, graph, h, edge_features):
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)  # , mod_kwargs=edge_features

            for key in h:
                if self.gnn_layer['combine'] == 'stack':
                    bs = h[key].shape[0]
                    h[key] = h[key].reshape(bs, -1)
                elif self.gnn_layer['combine'] == 'mean':
                    h[key] = th.mean(h[key], 1)

        return h

    def forward2(self, graph, inputs, edge_features):
        # inputs are features of nodes\
        h = inputs

        for i, layer in enumerate(self.layers):
            if edge_features:
                # h = layer(graph, h, mod_kwargs=edge_features)
                h = layer(graph, h)
            else:
                h = layer(graph, h)

            if self.gnn_layer['architecture'] == 'GATConv':
                for key in h:
                    if self.gnn_layer['combine'] == 'stack':
                        bs = h[key].shape[0]
                        h[key] = h[key].reshape(bs, -1)
                    elif self.gnn_layer['combine'] == 'mean':
                        h[key] = th.mean(h[key], 1)
                    elif self.gnn_layer['combine'] == 'max':
                        h[key] = th.max(h[key], 1).values
                    if self.gnn_layer['dropout']:
                        dropout = nn.Dropout(p=self.gnn_layer['dropout'])
                        h[key] = dropout(h[key])

            act = nn.PReLU()
            if i != len(self.layers) - 1:  # For all network layers but the last, add an activation function afterwards  -1

                h = {k: F.relu(v) for k, v in h.items()}  # relu, leaky_relu, mish

        return h

    def my_agg_func(self, tensors, dsttype):
        stacked = th.stack(tensors, dim=0)
        res = th.sum(stacked, dim=0)
        return res

    def diversity_regularization(self, tensors):
        reg = 0
        n_layers = len(tensors)
        for l in range(n_layers):
            attention_weights = tensors[l]
            n_heads = attention_weights.shape[1]
            for i in range(n_heads):
                for j in range(i + 1, n_heads):
                    ai = attention_weights[:, i, :]
                    aj = attention_weights[:, j, :]
                    cos_sim = F.cosine_similarity(ai, aj, dim=-1)
                    reg += cos_sim.mean()
        return reg

    def my_agg_func22(self, tensors, dst_type):
        tensors = [tup for k, tup in tensors.items()]
        self.div_reg = self.div_reg + self.diversity_regularization(tensors)

        if dst_type == 'drug':
            layer_weights = [1] * 5
            layer_weights = self.model_drug(th.tensor(layer_weights).to(th.float32).to('cuda'))
            self.drug_weigts = layer_weights
        else:
            layer_weights = [1] * 4
            layer_weights = self.model_target(th.tensor(layer_weights).to(th.float32).to('cuda'))
            self.target_weigts = layer_weights
        res = th.zeros(tensors[0].shape).to('cuda')
        for i, emb in enumerate(tensors):
            res = res + layer_weights[i] * emb
        return res

    def my_agg_func2(self, tensors, dst_type):
        tensors = [tup for k, tup in tensors.items()]

        if dst_type == 'drug':
            layer_weights = [1] * 5
            layer_weights = self.model_drug(th.tensor(layer_weights).to(th.float32).to('cuda'))
            self.drug_weigts = layer_weights
        else:
            layer_weights = [1] * 4
            layer_weights = self.model_target(th.tensor(layer_weights).to(th.float32).to('cuda'))
            self.target_weigts = layer_weights
        res = th.zeros(tensors[0].shape).to('cuda')
        for i, emb in enumerate(tensors):
            res = res + layer_weights[i] * emb
        return res

    def my_agg_func_conv(self, tensors, dst_type):
        tensors = [tup for k, tup in tensors.items()]
        stacked = th.stack(tensors, dim=1)
        if dst_type == 'drug':
            layer_weights = self.model_drug(stacked).squeeze()
        else:
            layer_weights = self.model_target(stacked).squeeze()
        return layer_weights

    def my_agg_func3(self, tensors, dsttype):
        tensors = [tup for k, tup in tensors.items()]
        stacked_node_feats = th.stack(tensors, dim=2)
        stacked = stacked_node_feats.flatten(start_dim=1, end_dim=2)

        if stacked_node_feats.shape[1] == self.hid_feats:
            if dsttype == 'drug':
                res, weights = self.attention_drugs(stacked, stacked, stacked)
                res = nn.functional.relu(res)
                res = self.att_linear_drugs(res)
            else:
                res, weights = self.attention_targets(stacked, stacked, stacked)
                res = nn.functional.relu(res)
                res = self.att_linear_targets(res)

        if stacked_node_feats.shape[1] == self.out_feats:
            if dsttype == 'drug':
                res, weights = self.attention_drugs2(stacked, stacked, stacked)
                res = nn.functional.relu(res)
                res = self.att_linear_drugs2(res)
            else:
                res, weights = self.attention_targets2(stacked, stacked, stacked)
                res = nn.functional.relu(res)
                res = self.att_linear_targets2(res)

        return res.to('cuda')

    def my_agg_test(self, tensors, dst_type):
        tensors = [tup for k, tup in tensors.items()]
        stacked_node_feats = th.stack(tensors, dim=2)
        stacked = stacked_node_feats.flatten(start_dim=1, end_dim=2)
        res = stacked.flatten(start_dim=1, end_dim=2)

        if dst_type == 'drug':
            if res.shape[1] == 900:
                res = self.att_linear_drugs4(res)
            else:
                res = self.att_linear_drugs3(res)
        else:
            if res.shape[1] == 720:
                res = self.att_linear_targets4(res)
            else:
                res = self.att_linear_targets3(res)
        return res

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, gnn_layer, n_layers, model_drug,
                 model_target):
        super().__init__()
        self.gnn = RGCN(in_features, hidden_features, out_features, rel_names, gnn_layer, n_layers, model_drug,
                        model_target)
        # self.pred = HeteroDotProductPredictor()

    def forward(self, g, x, edge_features):
        h = self.gnn(g, x, edge_features)
        return h

class AUPR_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(AUPR_loss, self).__init__()
        self.n_b = 21
        self._d = 0.05
        self._n1 = 1713

    def _f_delta(self, Y, h):
        # δ(Y^,h)
        temp = 1 - np.abs(Y - self._b[h]) / self._d
        return np.maximum(temp, 0)

    def forward(self, inputs, targets):

        Y_pred = self._get_prediction_trainset()
        yp_max = np.amax(Y_pred)
        yp_min = np.amin(Y_pred)
        h_min = int(self.n_b - 1 - int(yp_max / self._d + 1))  # yp_max=0.61 ==> h_min=7, self._b[h_min]=0.65
        h_min = max(h_min, 0)
        h_max = int(self.n_b - 1 - int(yp_min / self._d - 1))  # yp_min=0.36 ==> h_max-1=13, self._b[h_max-1]=0.35
        h_max = min(self.n_b, h_max)
        h_range = range(h_min, h_max)

        """compute ψ_h and bar_ψ_h """
        psi = np.zeros(self.n_b, dtype=float)
        psi_ = np.zeros(self.n_b, dtype=float)
        for h in h_range:
            X = self._f_delta(Y_pred, h)  # δ(Y^,h)
            """!!!New only training pairs are counted"""
            X = X * self._Omega
            psi[h] = np.sum(X * self._Y)  # ψ[δ(Y^,h)⊙Y]
            psi_[h] = np.sum(X)  # ψ[δ(Y^,h)]
        sum_psi = sum_psi_ = 0
        ap = 0
        for h in h_range:
            if psi_[h] == 0:
                continue
            else:
                sum_psi += psi[h]
                sum_psi_ += psi_[h]
                prec = sum_psi / sum_psi_
                recall = psi[h] / self._n1
                ap += prec * recall
        aupr_loss = (1 - ap) * self._n1  # for the acutual loss times self._n1

        return aupr_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(), targets.float())
        loss = self.alpha * (1 - th.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def focalLoss3(inputs, targets, alpha=1, gamma=1):
    BCE = F.binary_cross_entropy(input=inputs, target=targets)
    BCE_EXP = th.exp(-BCE)
    focal_loss = th.mean(alpha * th.pow((1 - BCE_EXP), gamma) * BCE)
    return focal_loss


def dual_focal_loss(scores, targets, a=1, b=1, r=1):
    focal_loss = 0
    for s, t in zip(scores, targets):
        term1 = t * th.log2(s)
        term2 = b * (1 - t) * th.log2((r - s))
        term3 = a * th.abs(t - s)
        focal_loss = term1 + term2 + term3
    return -focal_loss


def f_delta(Y, h):
    n_b = 21
    d = 1.0 / (n_b - 1)
    b = 1 - np.arange(n_b) * d
    # b = th.tensor(b)
    temp = 1 - th.abs(Y - b[h]) / d
    return th.maximum(temp, th.tensor([0]).to(temp.device))


def aupr_loss(scores, targets):
    n1 = th.sum(targets)
    yp_max = th.max(scores)
    yp_min = th.min(scores)
    n_b = 21
    d = 1.0 / (n_b - 1)
    h_min = int(n_b - 1 - int(yp_max / d + 1))
    h_min = max(h_min, 0)
    h_max = int(n_b - 1 - int(yp_min / d - 1))
    h_max = min(n_b, h_max)
    h_range = range(h_min, h_max)

    psi = th.tensor(np.zeros(n_b, dtype=float))
    psi_ = th.tensor(np.zeros(n_b, dtype=float))

    for h in h_range:
        X = f_delta(scores, h)

        psi[h] = th.sum(X * targets)
        psi_[h] = th.sum(X)
    sum_psi = sum_psi_ = 0
    ap = 0
    for h in h_range:
        if psi_[h] == 0:
            continue
        else:
            sum_psi = sum_psi + psi[h]
            sum_psi_ = sum_psi_ + psi_[h]
            prec = sum_psi / sum_psi_
            recall = psi[h] / n1
            ap = ap + prec * recall
    l_ap = (1 - ap) * n1

    return l_ap


def bpr_loss(pred_scores, true_scores):
    positive_scores = pred_scores[true_scores == 1].to('cpu')
    negative_scores = pred_scores[true_scores == 0].to('cpu')
    loss = -th.mean(th.log(th.sigmoid(positive_scores - th.unsqueeze(negative_scores, dim=1))))

    return loss.to('cuda')


def gae_loss(pos_score, neg_score, loss_type, device):
    scores = th.cat([pos_score, neg_score]).squeeze()
    targets = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).to(device)
    # targets = th.cat([th.ones(pos_score.shape[0])*0.95, th.zeros(neg_score.shape[0]) + 0.05]).to(device) # label smoothing
    if loss_type == 'bce':
        # loss = F.binary_cross_entropy(input=th.sigmoid(scores), target=targets)
        loss = F.binary_cross_entropy_with_logits(scores, targets)
    elif loss_type == 'wbce':
        pos_weight = neg_score.shape[0] / pos_score.shape[0]
        loss = F.binary_cross_entropy_with_logits(scores, targets, pos_weight=th.tensor([pos_weight]).to(device))
    elif loss_type == 'fl':
        a = neg_score.shape[0] / (neg_score.shape[0] + pos_score.shape[0])
        # loss = sfl(scores, targets, alpha=a, gamma=2, reduction='sum')
    elif loss_type == 'fl2':
        fl = FocalLoss()
        loss = fl(th.sigmoid(scores), targets)
    elif loss_type == 'fl3':
        loss = focalLoss3(th.sigmoid(scores), targets)
    elif loss_type == 'dfl':
        loss = dual_focal_loss(th.sigmoid(scores), targets)
    elif loss_type == 'aupr_loss':
        loss = aupr_loss(th.sigmoid(scores), targets)
    elif loss_type == 'aupr_wbce_loss':
        au_loss = aupr_loss(th.sigmoid(scores), targets)
        pos_weight = neg_score.shape[0] / pos_score.shape[0]
        wbce_loss = F.binary_cross_entropy_with_logits(scores, targets, pos_weight=th.tensor([pos_weight]).to(device))
        loss = au_loss / (pos_score.shape[0] / 2) + wbce_loss
    elif loss_type == 'BPR_loss':
        loss = bpr_loss(th.sigmoid(scores), targets)
    return loss


def get_preds_and_labels(pos_preds, neg_preds):
    scores = th.cat([pos_preds, neg_preds]).squeeze().detach()
    labels = th.cat([th.ones(pos_preds.shape[0]), th.zeros(neg_preds.shape[0])]).numpy()
    return scores, labels


def evaluate(preds, scores, labels, round_digits):
    fpr, tpr, thr = roc_curve(labels, scores)
    auc_score = round(auc(fpr, tpr), round_digits)
    aupr = round(average_precision_score(labels, scores), round_digits)

    f1 = round(f1_score(labels, preds, average='macro'), round_digits)
    acc = round(balanced_accuracy_score(labels, preds), round_digits)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    prec_pos = tp / (tp + fp)
    prec_neg = tn / (tn + fn)

    rec_pos = tp / (tp + fn)
    rec_neg = tn / (tn + fp)

    f1_pos = (2 * prec_pos * rec_pos) / (prec_pos + rec_pos)
    f1_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg)

    return auc_score, aupr, f1, acc, f1_pos, f1_neg, prec_pos, prec_neg, rec_pos, rec_neg


def print_evaluation_report_per_fold(history, fold):
    best_val_epoch_auc = np.argmax(history['val_auc'])
    val_epoch_auc_score = history['val_auc'][best_val_epoch_auc]
    test_epoch_auc_score = history['test_auc'][best_val_epoch_auc]

    best_val_epoch_aupr = np.argmax(history['val_aupr'])
    val_epoch_aupr_score = history['val_aupr'][best_val_epoch_aupr]
    test_epoch_aupr_score = history['test_aupr'][best_val_epoch_aupr]

    # print('-------------------------------------------------------------------------------------------------')
    print('Fold {}, best validation epoch is {}, with val AUC: {:.4f} and test AUC: {:.4f}'.format(
        fold, best_val_epoch_auc, val_epoch_auc_score, test_epoch_auc_score))
    print('Fold {}, best validation epoch is {}, with val AUPR: {:.4f} and test AUPR: {:.4f}'.format(
        fold, best_val_epoch_aupr, val_epoch_aupr_score, test_epoch_aupr_score))
    print('-------------------------------------------------------------------------------------------------')

    dir = 'figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    visualize_evaluation(history, best_val_epoch_auc, val_epoch_auc_score, best_val_epoch_aupr, val_epoch_aupr_score,
                         dir)


def visualize_evaluation(history, best_val_epoch_auc, val_epoch_auc_score, best_val_epoch_aupr, val_epoch_aupr_score,
                         dir):
    # Auc visualization
    plt.plot(history['train_auc'])
    plt.plot(history['val_auc'])
    plt.plot(history['test_auc'])
    plt.plot(best_val_epoch_auc, val_epoch_auc_score, 'yo')
    plt.title('Model ' + 'AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.savefig(dir + 'auc.png')
    plt.clf()

    # AUPR visualization
    plt.plot(history['train_aupr'])
    plt.plot(history['val_aupr'])
    plt.plot(history['test_aupr'])
    plt.plot(best_val_epoch_aupr, val_epoch_aupr_score, 'yo')
    plt.title('Model ' + 'AUPR')
    plt.ylabel('AUPR')
    plt.xlabel('epoch')
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.savefig(dir + 'aupr.png')
    plt.clf()

    # Per class F1 visualizations
    plt.plot(history['train_pos_f1'])
    plt.plot(history['train_neg_f1'])
    plt.title('Per class train F1 Score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['train_pos', 'train_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_train_f1.png')
    plt.clf()

    plt.plot(history['val_pos_f1'])
    plt.plot(history['val_neg_f1'])
    plt.title('Per class validation F1 Score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['val_pos', 'val_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_val_f1.png')
    plt.clf()

    plt.plot(history['test_pos_f1'])
    plt.plot(history['test_neg_f1'])
    plt.title('Per class test F1 Score')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['test_pos', 'test_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_test_f1.png')
    plt.clf()

    # Per class Precision visualizations
    plt.plot(history['train_pos_prec'])
    plt.plot(history['train_neg_prec'])
    plt.title('Per class train precision Score')
    plt.ylabel('Prec')
    plt.xlabel('epoch')
    plt.legend(['train_pos', 'train_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_train_prec.png')
    plt.clf()

    plt.plot(history['val_pos_prec'])
    plt.plot(history['val_neg_prec'])
    plt.title('Per class validation precision Score')
    plt.ylabel('Prec')
    plt.xlabel('epoch')
    plt.legend(['val_pos', 'val_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_val_prec.png')
    plt.clf()

    plt.plot(history['test_pos_prec'])
    plt.plot(history['test_neg_prec'])
    plt.title('Per class test precision Score')
    plt.ylabel('Prec')
    plt.xlabel('epoch')
    plt.legend(['test_pos', 'test_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_test_prec.png')
    plt.clf()

    # Per class Recall visualizations
    plt.plot(history['train_pos_rec'])
    plt.plot(history['train_neg_rec'])
    plt.title('Per class train recall Score')
    plt.ylabel('Rec')
    plt.xlabel('epoch')
    plt.legend(['train_pos', 'train_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_train_rec.png')
    plt.clf()

    plt.plot(history['val_pos_rec'])
    plt.plot(history['val_neg_rec'])
    plt.title('Per class validation recall Score')
    plt.ylabel('Rec')
    plt.xlabel('epoch')
    plt.legend(['val_pos', 'val_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_val_rec.png')
    plt.clf()

    plt.plot(history['test_pos_rec'])
    plt.plot(history['test_neg_rec'])
    plt.title('Per class test recall Score')
    plt.ylabel('Rec')
    plt.xlabel('epoch')
    plt.legend(['test_pos', 'test_neg'], loc='upper left')
    plt.savefig(dir + 'per_class_test_rec.png')
    plt.clf()


def visualize_evaluation_all_folds(history):
    n_epochs = len(history[0]['train_auc'])
    n_folds = len(history)
    avg_metrics = {
        'train_auc': [0] * n_epochs,
        'val_auc': [0] * n_epochs,
        'test_auc': [0] * n_epochs,

        'train_aupr': [0] * n_epochs,
        'val_aupr': [0] * n_epochs,
        'test_aupr': [0] * n_epochs,

        'train_pos_f1': [0] * n_epochs,
        'train_neg_f1': [0] * n_epochs,

        'val_pos_f1': [0] * n_epochs,
        'val_neg_f1': [0] * n_epochs,

        'test_pos_f1': [0] * n_epochs,
        'test_neg_f1': [0] * n_epochs,

        'train_pos_prec': [0] * n_epochs,
        'train_neg_prec': [0] * n_epochs,

        'val_pos_prec': [0] * n_epochs,
        'val_neg_prec': [0] * n_epochs,

        'test_pos_prec': [0] * n_epochs,
        'test_neg_prec': [0] * n_epochs,

        'train_pos_rec': [0] * n_epochs,
        'train_neg_rec': [0] * n_epochs,

        'val_pos_rec': [0] * n_epochs,
        'val_neg_rec': [0] * n_epochs,

        'test_pos_rec': [0] * n_epochs,
        'test_neg_rec': [0] * n_epochs
    }

    for fold in history:
        for epoch in range(n_epochs):
            avg_metrics['train_auc'][epoch] += fold['train_auc'][epoch] / n_folds
            avg_metrics['val_auc'][epoch] += fold['val_auc'][epoch] / n_folds
            avg_metrics['test_auc'][epoch] += fold['test_auc'][epoch] / n_folds

            avg_metrics['train_aupr'][epoch] += fold['train_aupr'][epoch] / n_folds
            avg_metrics['val_aupr'][epoch] += fold['val_aupr'][epoch] / n_folds
            avg_metrics['test_aupr'][epoch] += fold['test_aupr'][epoch] / n_folds

            avg_metrics['train_pos_f1'][epoch] += fold['train_pos_f1'][epoch] / n_folds
            avg_metrics['train_neg_f1'][epoch] += fold['train_neg_f1'][epoch] / n_folds

            avg_metrics['val_pos_f1'][epoch] += fold['val_pos_f1'][epoch] / n_folds
            avg_metrics['val_neg_f1'][epoch] += fold['val_neg_f1'][epoch] / n_folds

            avg_metrics['test_pos_f1'][epoch] += fold['test_pos_f1'][epoch] / n_folds
            avg_metrics['test_neg_f1'][epoch] += fold['test_neg_f1'][epoch] / n_folds

            avg_metrics['train_pos_prec'][epoch] += fold['train_pos_prec'][epoch] / n_folds
            avg_metrics['train_neg_prec'][epoch] += fold['train_neg_prec'][epoch] / n_folds

            avg_metrics['val_pos_prec'][epoch] += fold['val_pos_prec'][epoch] / n_folds
            avg_metrics['val_neg_prec'][epoch] += fold['val_neg_prec'][epoch] / n_folds

            avg_metrics['test_pos_prec'][epoch] += fold['test_pos_prec'][epoch] / n_folds
            avg_metrics['test_neg_prec'][epoch] += fold['test_neg_prec'][epoch] / n_folds

            avg_metrics['train_pos_rec'][epoch] += fold['train_pos_rec'][epoch] / n_folds
            avg_metrics['train_neg_rec'][epoch] += fold['train_neg_rec'][epoch] / n_folds

            avg_metrics['val_pos_rec'][epoch] += fold['val_pos_rec'][epoch] / n_folds
            avg_metrics['val_neg_rec'][epoch] += fold['val_neg_rec'][epoch] / n_folds

            avg_metrics['test_pos_rec'][epoch] += fold['test_pos_rec'][epoch] / n_folds
            avg_metrics['test_neg_rec'][epoch] += fold['test_neg_rec'][epoch] / n_folds


def generate_confusion_matrix(y_test, y_test_pred, test_scores):
    if test_scores[-1] == max(
            test_scores):  # Save the new confusion matrix only if the current metric is the highest so far
        dir = 'figures/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(dir + "confusion_matrix_plot.png"):
            os.remove(dir + "confusion_matrix_plot.png")
        cf_matrix = confusion_matrix(y_test, y_test_pred)

        group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
        # group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        n_ones = np.sum(y_test)
        n_zeros = len(y_test) - n_ones
        group_percentages = ['{0:.2%}'.format(int(cf_matrix.flatten()[0]) / n_zeros), '-', '-',
                             '{0:.2%}'.format(int(cf_matrix.flatten()[3]) / n_ones)]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        n_classes = len(Counter(y_test.tolist()))
        labels = np.asarray(labels).reshape(n_classes, n_classes)
        # sns.set(rc={'figure.figsize':(8,6)})
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        # plt.show()
        plt.savefig(dir + 'confusion_matrix_plot.png')
        plt.clf()


def print_evaluation_report_all_folds(folds_history, metric, round_digits):
    folds_best_ep_hist = []
    for fold_i, history in enumerate(folds_history):
        best_val_epoch = np.argmax(history['val_' + metric])
        best_val_epoch_score = history['val_' + metric][best_val_epoch]
        best_test_epoch_score = history['test_' + metric][best_val_epoch]
        best_epoch_history = {
            'val_' + metric: best_val_epoch_score,
            'test_' + metric: best_test_epoch_score,
            'epoch': best_val_epoch
        }
        folds_best_ep_hist.append(best_epoch_history)

    max_val_score = 0
    max_test_score = 0
    average_val_score = 0
    average_test_score = 0

    for fold_i, history in enumerate(folds_best_ep_hist):
        average_val_score += history['val_' + metric] / len(folds_best_ep_hist)
        average_test_score += history['test_' + metric] / len(folds_best_ep_hist)
        max_val_score = max(max_val_score, history['val_' + metric])
        max_test_score = max(max_test_score, history['test_' + metric])
    print('{} -- Average best val score: {}, average test score: {}. Max val score: {}, test score: {}'.format(metric,
                                                                                                               round(
                                                                                                                   average_val_score,
                                                                                                                   round_digits),
                                                                                                               round(
                                                                                                                   average_test_score,
                                                                                                                   round_digits),
                                                                                                               max_val_score,
                                                                                                               max_test_score))


def load_data(file_dir, data_type, type='np', divide_with=1, sep='\t'):
    if type == 'np':
        mat = np.loadtxt(file_dir, dtype=str)
        mat = mat.astype(float) if data_type == 'float' else mat.astype(int)
        mat = mat / divide_with
    elif type == 'pd':
        mat = pd.read_csv(file_dir, sep=sep, header=None).to_numpy()[1:, 1:].astype(float)
        mat = mat / divide_with

    u, v = from_matrix_to_edge_list(mat, mode=data_type)
    u_v, u_v_reverse = from_edge_list_to_tuple_list(u, v)
    u_v_mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))  # Normalize matrix between 0-1
    return u_v, u_v_reverse, u_v_mat

def find_k_similar(file_dir, k, file_type='np', real_value=False, devide_with=1):
    if file_type == 'np':
        mat = np.loadtxt(file_dir, dtype=str)
        mat = mat.astype(float)
    elif file_type == 'pd':
        mat = pd.read_csv(file_dir, sep='\t', header=None).to_numpy()[1:, 1:].astype(float)

    mat = mat / devide_with
    mat_knn = np.copy(mat)

    k = k + 1  # find k+1 most similar drugs to each drug, as we will remove its self from the list of most similar
    for i in range(mat_knn.shape[0]):
        indices = np.argpartition(mat_knn[i], -k)[-k:]
        for j in range(mat_knn.shape[1]):
            if real_value:
                if j in indices and j != i:
                    pass
                else:
                    mat_knn[i, j] = 0
            else:
                if j in indices and j != i:
                    mat_knn[i, j] = 1
                else:
                    mat_knn[i, j] = 0

    if real_value:
        u, v = from_matrix_to_edge_list(mat_knn, mode='float', include_zeros=False)
    else:
        u, v = from_matrix_to_edge_list(mat_knn, mode='int')
    u_v, u_v_reverse = from_edge_list_to_tuple_list(u, v)
    # u_v_mat = (mat_knn - np.min(mat_knn)) / (np.max(mat_knn) - np.min(mat_knn))  # Normalize matrix between 0-1
    return u_v, u_v_reverse, mat_knn, mat


def find_weighted_k_similar_from_matrix(matrix, k):
    k = k + 1  # find k+1 most similar drugs to each drug, as we will remove its self from the list of most similar
    weigted_KNN = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        indices = np.argsort(matrix[i])[-k:][::-1]
        indices = indices[indices != i]
        for lst_idx, j in enumerate(indices):

            if lst_idx == 0:
                weigted_KNN[i, j] = 1 + 1
            elif lst_idx == 1:
                weigted_KNN[i, j] = 1.5
            elif lst_idx == 2:
                weigted_KNN[i, j] = 1.25
            weigted_KNN[i, i] = 2


    u, v = from_matrix_to_edge_list(weigted_KNN, mode='float', include_zeros=False)
    u_v, u_v_reverse = from_edge_list_to_tuple_list(u, v)
    weigted_KNN = weigted_KNN.flatten()
    weigted_KNN = weigted_KNN[weigted_KNN != 0]
    return u_v, u_v_reverse, weigted_KNN


def save_predictions(history, metric='val_aupr'):
    dir = 'best_predictions/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    best_val_epoch = np.argmax(history[metric])

    val_preds = history['val_preds'][best_val_epoch]
    val_probs = history['val_probs'][best_val_epoch]
    val_labels = history['val_labels'][best_val_epoch]

    test_preds = history['test_preds'][best_val_epoch]
    test_probs = history['test_probs'][best_val_epoch]
    test_labels = history['test_labels'][best_val_epoch]

    file1 = open(dir + "best_epoch_val_preds.txt", "w")
    val_preds = [str(el) for el in val_preds]
    file1.write(','.join(val_preds))
    file1.close()

    file1 = open(dir + "best_epoch_val_probs.txt", "w")
    val_probs = [str(el) for el in val_probs]
    file1.write(','.join(val_probs))
    file1.close()

    file1 = open(dir + "best_epoch_val_labels.txt", "w")
    val_labels = [str(el) for el in val_labels]
    file1.write(','.join(val_labels))
    file1.close()

    file1 = open(dir + "best_epoch_test_preds.txt", "w")
    test_preds = [str(el) for el in test_preds]
    file1.write(','.join(test_preds))
    file1.close()

    file1 = open(dir + "best_epoch_test_probs.txt", "w")
    test_probs = [str(el) for el in test_probs]
    file1.write(','.join(test_probs))
    file1.close()

    file1 = open(dir + "best_epoch_test_labels.txt", "w")
    test_labels = [str(el) for el in test_labels]
    file1.write(','.join(test_labels))
    file1.close()


def standardize(arr):
    arr = arr - np.eye(arr.shape[0])
    mu = arr.mean()
    sigma = statistics.stdev(arr.flatten())
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = (arr[i, j] - mu) / sigma
    return arr


def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] != 0:
                arr[i, j] = (arr[i, j] - arr_min) / (arr_max - arr_min)
    return arr


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(th.randn(1)) for i in range(n_inputs)])

    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res


def check_early_stopping(history, patience):
    current_score = history['val_aupr'][-1]
    best_score_epoch = np.argmax(history['val_aupr']) + 1
    # best_score = history['val_aupr'][best_score_epoch-1]
    current_epoch = len(history['val_aupr'])

    patience += 1
    if current_epoch > patience:
        if current_score < min(history['val_aupr'][-patience:][:-1]):
            return True
    else:
        return False


def create_combined_interaction_matrix(pos_graph, pos_scores, neg_graph, neg_scores, threshold, edge_type):
    num_drugs = 708
    num_targets = 1512

    interaction_matrix = th.zeros((num_drugs, num_targets))

    # Get the indices of the drug-target pairs for positive and negative edges
    pos_src, pos_dst = pos_graph.edges(etype=edge_type, order='eid')
    neg_src, neg_dst = neg_graph.edges(etype=edge_type, order='eid')

    # Apply threshold to determine positive interactions
    pos_mask = pos_scores.squeeze() > threshold
    neg_mask = neg_scores.squeeze() > threshold

    interaction_matrix[neg_src[neg_mask], neg_dst[neg_mask]] = neg_scores.squeeze()[neg_mask].cpu()

    return interaction_matrix.numpy()


def run_experiment(model, interaction_data, setting, gnn_layer, n_layers, feat_type, feat_size, hidden_size, emb_size,
                   self_loops, neg_ratio, n_folds, lr, loss,
                   n_epochs, verbose, round_digits, net_layers, similarity_layer_KNN, affinity_layer_integer, device):
    if device == 'auto':
        device = 'cuda' if th.cuda.is_available() else 'cpu'
    device = th.device(device)

    print(
        '\n\n--- Experiment details \n model: {}\n interaction_data: {}\n setting: {}\n gnn layer: {}\n n_layers: {}\n feat_type: {}\n feat_size: {}\n hidden_size: {}\n emb_size: {}\n self_loops: {}\n neg_ratio: {}\n lr: {}\n loss: {}\n n_epochs: {}\n n_folds: {}\n net_layers: {}\n similarity_layer_KNN: {}\n affinity_layer_integer: {}\n device: {}\n'.format(
            model, interaction_data, setting, gnn_layer, n_layers, feat_type, feat_size, hidden_size, emb_size,
            self_loops, neg_ratio, lr, loss, n_epochs,
            n_folds, net_layers, similarity_layer_KNN, affinity_layer_integer, device))

    luo_interactions = np.loadtxt("data/mat_drug_protein_{}.txt".format(interaction_data), dtype=str)
    luo_interactions = luo_interactions.astype(int)

    u, v = from_matrix_to_edge_list(luo_interactions)

    if n_folds == 1:
        folds = train_val_test_split(u, v, seed)
    else:
        folds = cross_val_split(luo_interactions, setting, n_folds, 0.1, neg_ratio, seed)

    drug_drug_int, _, _ = load_data("data/mat_drug_drug.txt", 'int')
    target_target_int, _, _ = load_data("data/mat_protein_protein.txt", 'int')
    drug_se_assoc, drug_se_assoc_reverse, _ = load_data("data/mat_drug_se.txt", 'int')
    target_dis_assoc, target_dis_assoc_reverse, _ = load_data("data/mat_protein_disease.txt", 'int')

    edge_info = {}
    if similarity_layer_KNN:
        drug_drug_similarity, _, _, _ = find_k_similar("data/Similarity_Matrix_Drugs.txt", 3)
        target_target_similarity, _, _, _ = find_k_similar("data/Similarity_Matrix_Proteins.txt", 3)

        drug_drug_similarity_2, _, drug_drug_similarity_2_mat, _ = find_k_similar(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_ddi.txt", 3, 'pd')
        drug_drug_similarity_3, _, drug_drug_similarity_3_mat, _ = find_k_similar(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_disease.txt", 3, 'pd')
        drug_drug_similarity_4, _, drug_drug_similarity_4_mat, _ = find_k_similar(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_se.txt", 3, 'pd')
        drug_drug_similarity_5, _, drug_drug_similarity_5_mat, _ = find_k_similar(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_tanimoto.txt", 3, 'pd')

        target_target_similarity_2, _, target_target_similarity_2_mat, _ = find_k_similar(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_disease.txt", 3, 'pd')
        target_target_similarity_3, _, target_target_similarity_3_mat, _ = find_k_similar(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_ppi.txt", 3, 'pd')
        target_target_similarity_4, _, target_target_similarity_4_mat, _ = find_k_similar(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_sw-n.txt", 3, 'pd')

    else:
        drug_drug_similarity, _, drug_drug_mat = load_data("data/Similarity_Matrix_Drugs.txt", 'float', type='np')

        target_target_similarity, _, target_target_mat = load_data("data/Similarity_Matrix_Proteins.txt", 'float',
                                                                   type='np', divide_with=100)


        drug_drug_similarity, _, drug_drug_mat = find_weighted_k_similar_from_matrix(drug_drug_mat, 3)
        target_target_similarity, _, target_target_mat = find_weighted_k_similar_from_matrix(target_target_mat, 3)

        edge_info['is_similar_drug'] = drug_drug_mat
        edge_info['is_similar_target'] = target_target_mat

        drug_drug_similarity_2, _, drug_drug_similarity_2_mat = load_data(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_ddi.txt", 'float', 'pd')
        drug_drug_similarity_3, _, drug_drug_similarity_3_mat = load_data(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_disease.txt", 'float', 'pd')
        drug_drug_similarity_4, _, drug_drug_similarity_4_mat = load_data(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_se.txt", 'float', 'pd')
        drug_drug_similarity_5, _, drug_drug_similarity_5_mat = load_data(
            "data/MDMF2_similarities_drugs/MDMF2_luo_simmat_drugs_tanimoto.txt", 'float', 'pd')

        target_target_similarity_2, _, target_target_2_mat = load_data(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_disease.txt", 'float', 'pd')
        target_target_similarity_3, _, target_target_3_mat = load_data(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_ppi.txt", 'float', 'pd')
        target_target_similarity_4, _, target_target_4_mat = load_data(
            "data/MDMF2_similarities_targets/MDMF2_luo_simmat_proteins_sw-n.txt", 'float', 'pd')



        drug_drug_similarity_2, _, drug_drug_similarity_2_mat = find_weighted_k_similar_from_matrix(
            drug_drug_similarity_2_mat, 3)
        drug_drug_similarity_3, _, drug_drug_similarity_3_mat = find_weighted_k_similar_from_matrix(
            drug_drug_similarity_3_mat, 3)
        drug_drug_similarity_4, _, drug_drug_similarity_4_mat = find_weighted_k_similar_from_matrix(
            drug_drug_similarity_4_mat, 3)


        target_target_similarity_2, _, target_target_2_mat = find_weighted_k_similar_from_matrix(target_target_2_mat, 3)
        target_target_similarity_3, _, target_target_3_mat = find_weighted_k_similar_from_matrix(target_target_3_mat, 3)

        edge_info['is_similar_drug_2'] = drug_drug_similarity_2_mat
        edge_info['is_similar_drug_3'] = drug_drug_similarity_3_mat
        edge_info['is_similar_drug_4'] = drug_drug_similarity_4_mat
        # edge_info['is_similar_drug_5'] = drug_drug_similarity_5_mat

        edge_info['is_similar_target_2'] = target_target_2_mat
        edge_info['is_similar_target_3'] = target_target_3_mat
        # edge_info['is_similar_target_4'] = target_target_4_mat


    folds_history = []
    start_train_time = time.time()
    for fold_i, fold in enumerate(folds):

        train_pos_u, train_pos_v = fold['train_set_pos']
        train_neg_u, train_neg_v = fold['train_set_neg']

        val_pos_u, val_pos_v = fold['val_set_pos']
        val_neg_u, val_neg_v = fold['val_set_neg']

        test_pos_u, test_pos_v = fold['test_set_pos']
        test_neg_u, test_neg_v = fold['test_set_neg']


        num_nodes_dict = {
            'drug': 708,
            'target': 1512,
        }

        drug_target_int, drug_target_reverse = from_edge_list_to_tuple_list(train_pos_u, train_pos_v)
        train_pos_edge_graph = {
            ('drug', 'interacts', 'target'): drug_target_int,
            ('target', 'interacts', 'drug'): drug_target_reverse
        }

        if ('drug', 'is_similar_drug', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'is_similar_drug', 'drug')] = drug_drug_similarity
        if ('target', 'is_similar_target', 'target') in net_layers:
            train_pos_edge_graph[('target', 'is_similar_target', 'target')] = target_target_similarity
        if ('drug', 'interacts_drug', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'interacts_drug', 'drug')] = drug_drug_int
        if ('target', 'interacts_target', 'target') in net_layers:
            train_pos_edge_graph[('target', 'interacts_target', 'target')] = target_target_int
        if ('drug', 'associates', 'side_effect') in net_layers:
            train_pos_edge_graph[('drug', 'associates', 'side_effect')] = drug_se_assoc
            train_pos_edge_graph[('side_effect', 'associates', 'drug')] = drug_se_assoc_reverse
            if 'side_effect' not in feat_type:
                feat_type['side_effect'] = 'random'
        if ('target', 'causes', 'disease') in net_layers:
            train_pos_edge_graph[('target', 'causes', 'disease')] = target_dis_assoc
            train_pos_edge_graph[('disease', 'is_caused_by', 'target')] = target_dis_assoc_reverse
            if 'disease' not in feat_type:
                feat_type['disease'] = 'random'

        if ('drug', 'is_similar_drug_2', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'is_similar_drug_2', 'drug')] = drug_drug_similarity_2
        if ('drug', 'is_similar_drug_3', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'is_similar_drug_3', 'drug')] = drug_drug_similarity_3
        if ('drug', 'is_similar_drug_4', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'is_similar_drug_4', 'drug')] = drug_drug_similarity_4
        if ('drug', 'is_similar_drug_5', 'drug') in net_layers:
            train_pos_edge_graph[('drug', 'is_similar_drug_5', 'drug')] = drug_drug_similarity_5

        if ('target', 'is_similar_target_2', 'target') in net_layers:
            train_pos_edge_graph[('target', 'is_similar_target_2', 'target')] = target_target_similarity_2
        if ('target', 'is_similar_target_3', 'target') in net_layers:
            train_pos_edge_graph[('target', 'is_similar_target_3', 'target')] = target_target_similarity_3
        if ('target', 'is_similar_target_4', 'target') in net_layers:
            train_pos_edge_graph[('target', 'is_similar_target_4', 'target')] = target_target_similarity_4


        train_pos_edge_graph = dgl.heterograph(train_pos_edge_graph, num_nodes_dict=num_nodes_dict)

        if self_loops:
            addSelfLoop = AddSelfLoop(allow_duplicate=False, new_etypes=True)
            train_pos_edge_graph = addSelfLoop(train_pos_edge_graph)

        set_node_features(train_pos_edge_graph, feat_size, feat_type)
        edge_features = set_edge_features(train_pos_edge_graph, edge_info, affinity_layer_integer)

        drug_target_int_val, drug_target_int_val_reverse = from_edge_list_to_tuple_list(val_pos_u, val_pos_v)
        val_pos_edge_graph = {
            ('drug', 'interacts', 'target'): drug_target_int_val,
            ('target', 'interacts', 'drug'): drug_target_int_val_reverse
        }
        val_pos_edge_graph = dgl.heterograph(val_pos_edge_graph, num_nodes_dict=num_nodes_dict)

        drug_target_int_test, drug_target_int_test_reverse = from_edge_list_to_tuple_list(test_pos_u, test_pos_v)
        test_pos_edge_graph = {
            ('drug', 'interacts', 'target'): drug_target_int_test,
            ('target', 'interacts', 'drug'): drug_target_int_test_reverse
        }
        test_pos_edge_graph = dgl.heterograph(test_pos_edge_graph, num_nodes_dict=num_nodes_dict)

        model_drug = nn.Sequential(
            nn.Linear(5, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 5),
            # nn.Softmax()
            nn.Sigmoid()
            # nn.ReLU(),
            # nn.Linear(400, 400)
        )
        model_target = nn.Sequential(
            nn.Linear(4, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 4),
            # nn.Softmax()
            nn.Sigmoid()
            # nn.ReLU(),
            # nn.Linear(400, 400)
        )


        drug_feats = train_pos_edge_graph.nodes['drug'].data['h']
        target_feats = train_pos_edge_graph.nodes['target'].data['h']
        node_features = {'drug': drug_feats,
                         'target': target_feats}  # , 'side_effect': side_effect_feats, 'disease': disease_feats
        if 'side_effect' in feat_type:
            side_effect_feats = train_pos_edge_graph.nodes['side_effect'].data['h']
            node_features['side_effect'] = side_effect_feats
        if 'disease' in feat_type:
            disease_feats = train_pos_edge_graph.nodes['disease'].data['h']
            node_features['disease'] = disease_feats

        if model == 'GAE':
            model = Model(feat_size, hidden_size, emb_size, train_pos_edge_graph.etypes, gnn_layer, n_layers,
                          model_drug, model_target).to(device)
            compute_loss = gae_loss


        opt = th.optim.Adam(list(model.parameters()) + list(model_drug.parameters()) + list(model_target.parameters()))  # , lr=5e-4


        etypes = [('drug', 'interacts', 'target'), ('target', 'interacts', 'drug')]
        etype = ('drug', 'interacts', 'target')
        pred = HeteroDotProductPredictor()

        history = {
            'pred_test_mat': [],
            'train_auc': [],
            'val_auc': [],
            'test_auc': [],
            'train_aupr': [],
            'val_aupr': [],
            'test_aupr': [],

            'train_pos_f1': [],
            'train_neg_f1': [],
            'val_pos_f1': [],
            'val_neg_f1': [],
            'test_pos_f1': [],
            'test_neg_f1': [],

            'train_pos_prec': [],
            'train_neg_prec': [],
            'val_pos_prec': [],
            'val_neg_prec': [],
            'test_pos_prec': [],
            'test_neg_prec': [],

            'train_pos_rec': [],
            'train_neg_rec': [],
            'val_pos_rec': [],
            'val_neg_rec': [],
            'test_pos_rec': [],
            'test_neg_rec': [],

            'val_preds': [],
            'val_probs': [],
            'val_labels': [],
            'test_preds': [],
            'test_probs': [],
            'test_labels': []
        }
        logits_list = []

        train_neg_edge_graph = construct_negative_graph(train_pos_edge_graph, etype, train_neg_u, train_neg_v)

        # val_neg_u, val_neg_v = sample_datapoints(val_neg_u, val_neg_v, val_pos_u, 10)  # Negative Sampling
        val_neg_edge_graph = construct_negative_graph(val_pos_edge_graph, etype, val_neg_u, val_neg_v)

        # test_neg_u, test_neg_v = sample_datapoints(test_neg_u, test_neg_v, test_pos_u, 10) # Negative Sampling
        test_neg_edge_graph = construct_negative_graph(test_pos_edge_graph, etype, test_neg_u, test_neg_v)

        train_pos_edge_graph = train_pos_edge_graph.to(device)
        train_neg_edge_graph = train_neg_edge_graph.to(device)
        val_pos_edge_graph = val_pos_edge_graph.to(device)
        val_neg_edge_graph = val_neg_edge_graph.to(device)
        test_pos_edge_graph = test_pos_edge_graph.to(device)
        test_neg_edge_graph = test_neg_edge_graph.to(device)
        for key in node_features:
            node_features[key] = node_features[key].to(device)
        for key in edge_features:
            edge_features[key]['edge_weight'] = edge_features[key]['edge_weight'].to(device)
        sigmoid = nn.Sigmoid()

        for epoch in range(n_epochs):

            # Training and generating train prediction scores and computing train loss
            model.train()
            # model.gnn.div_reg = 0
            if bool(edge_features):
                logits = model(train_pos_edge_graph, node_features, edge_features)  # edge_features, None
            else:
                logits = model(train_pos_edge_graph, node_features, None)  # edge_features, None

            # logits_list.append(logits)
            train_pos_scores = pred(train_pos_edge_graph, logits, etypes[0])
            train_neg_scores = pred(train_neg_edge_graph, logits, etypes[0])
            # train_pos_scores, train_neg_scores = pred2(logits, train_inter, train_neg_inter)

            train_loss = compute_loss(train_pos_scores, train_neg_scores, loss, device)  # + 0.001 * model.gnn.div_reg
            # train_loss = model.compute_loss(train_pos_scores, train_neg_scores, device)
            model.eval()

            with th.no_grad():
                # Generating validation prediction scores and computing validation loss
                val_pos_scores = pred(val_pos_edge_graph, logits, etypes[0])
                val_neg_scores = pred(val_neg_edge_graph, logits, etypes[0])

                val_loss = compute_loss(val_pos_scores, val_neg_scores, loss, device)

                # Generating test prediction scores and computing test loss
                test_pos_scores = pred(test_pos_edge_graph, logits, etypes[0])
                test_neg_scores = pred(test_neg_edge_graph, logits, etypes[0])

                test_loss = compute_loss(test_pos_scores, test_neg_scores, loss, device)

            # Get pred logits and true labels
            train_pred_scores, train_labels = get_preds_and_labels(train_pos_scores, train_neg_scores)
            val_pred_scores, val_labels = get_preds_and_labels(val_pos_scores, val_neg_scores)
            test_pred_scores, test_labels = get_preds_and_labels(test_pos_scores, test_neg_scores)

            # Generate binary predictions from scores
            train_probs = sigmoid(train_pred_scores).detach().cpu().numpy()
            val_probs = sigmoid(val_pred_scores).detach().cpu().numpy()
            test_probs = sigmoid(test_pred_scores).detach().cpu().numpy()

            train_preds = np.round(train_probs)
            val_preds = np.round(val_probs)
            test_preds = np.round(test_probs)


            pred_test_mat = create_combined_interaction_matrix(test_pos_edge_graph, test_pos_scores,
                                                               test_neg_edge_graph, test_neg_scores, 0.5, etypes[0])

            opt.zero_grad()
            train_loss.backward()
            opt.step()


            # Calculate metrics
            train_auc, train_aupr, train_f1, train_acc, train_pos_f1, train_neg_f1, train_pos_prec, train_neg_prec, train_pos_rec, train_neg_rec = evaluate(
                train_preds, train_probs, train_labels, round_digits)
            val_auc, val_aupr, val_f1, val_acc, val_pos_f1, val_neg_f1, val_pos_prec, val_neg_prec, val_pos_rec, val_neg_rec = evaluate(
                val_preds, val_probs, val_labels, round_digits)
            test_auc, test_aupr, test_f1, test_acc, test_pos_f1, test_neg_f1, test_pos_prec, test_neg_prec, test_pos_rec, test_neg_rec = evaluate(
                test_preds, test_probs, test_labels, round_digits)

            # scheduler.step(val_auc)

            history['pred_test_mat'].append(pred_test_mat)

            history['val_preds'].append(val_preds)
            history['val_probs'].append(val_probs)
            history['val_labels'].append(val_labels)

            history['test_preds'].append(test_preds)
            history['test_probs'].append(test_probs)
            history['test_labels'].append(test_labels)

            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            history['test_auc'].append(test_auc)

            history['train_aupr'].append(train_aupr)
            history['val_aupr'].append(val_aupr)
            history['test_aupr'].append(test_aupr)

            history['train_pos_f1'].append(train_pos_f1)
            history['train_neg_f1'].append(train_neg_f1)
            history['val_pos_f1'].append(val_pos_f1)
            history['val_neg_f1'].append(val_neg_f1)
            history['test_pos_f1'].append(test_pos_f1)
            history['test_neg_f1'].append(test_neg_f1)

            history['train_pos_prec'].append(train_pos_prec)
            history['train_neg_prec'].append(train_neg_prec)
            history['val_pos_prec'].append(val_pos_prec)
            history['val_neg_prec'].append(val_neg_prec)
            history['test_pos_prec'].append(test_pos_prec)
            history['test_neg_prec'].append(test_neg_prec)

            history['train_pos_rec'].append(train_pos_rec)
            history['train_neg_rec'].append(train_neg_rec)
            history['val_pos_rec'].append(val_pos_rec)
            history['val_neg_rec'].append(val_neg_rec)
            history['test_pos_rec'].append(test_pos_rec)
            history['test_neg_rec'].append(test_neg_rec)

            generate_confusion_matrix(test_labels, test_preds, history['val_auc'])

            cur_lr = 0

            if verbose == 3 and epoch % 1 == 0:
                print(
                    'In epoch {}, Loss - train: {}, val: {},   AUC - val: {}, test: {},   AUPR - val: {}, test: {},   F1 - val: {}, test: {},   Acc - val: {}, test: {},  cur_lr: {}'.format(
                        epoch, round(train_loss.item(), 3), round(val_loss.item(), 3), val_auc, test_auc, val_aupr,
                        test_aupr, val_f1, test_f1, val_acc, test_acc, cur_lr))

            # if check_early_stopping(history, 50):
            # break

        # save_predictions(history, metric='val_auc')
        folds_history.append(history)
        print_evaluation_report_per_fold(history, fold=fold_i)


    print_evaluation_report_all_folds(folds_history, 'auc', round_digits)
    print_evaluation_report_all_folds(folds_history, 'aupr', round_digits)
    visualize_evaluation_all_folds(folds_history)

    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    print('Total train time: {}'.format(round(train_time, 0)))


net_layers = [('drug', 'is_similar_drug', 'drug'),
              ('target', 'is_similar_target', 'target'),
              ('drug', 'is_similar_drug_2', 'drug'), ('drug', 'is_similar_drug_3', 'drug'),
              ('drug', 'is_similar_drug_4', 'drug'),
              ('target', 'is_similar_target_2', 'target'), ('target', 'is_similar_target_3', 'target'),
              ]

params = {
    'model': 'GAE',  # GAE, VGAE
    'interaction_data': 'original',  # original, updated
    'setting': 's1',
    'gnn_layer': {'architecture': 'GATConv', 'n_heads': 1, 'combine': 'stack', 'dropout': None},
    'n_layers': 2, 'feat_type': {'drug': 'loaded_luo', 'target': 'loaded_luo'},
    'feat_size': 400,
    'hidden_size': 180,
    'emb_size': 40,
    'self_loops': False,
    'neg_ratio': None,  # None, 10
    'n_folds': 10,
    'lr': 0.1,
    'loss': 'aupr_wbce_loss',  # bce, wbce, fl, test, BPR_loss, aupr_loss, auc_loss, aupr_wbce_loss
    'n_epochs': 10,  # 50, 250, 1250
    'verbose': 2,
    'round_digits': 4,
    'net_layers': net_layers,
    'similarity_layer_KNN': False,
    'affinity_layer_integer': False,
    'device': 'cuda'  # cpu, cuda, auto

}

seed = 3
set_seed(seed)

# get_interaction_drugbank('data')


run_experiment(**params)
# '''
params['setting'] = 's2'
run_experiment(**params)
params['setting'] = 's3'
run_experiment(**params)
params['setting'] = 's4'
run_experiment(**params)
# '''





