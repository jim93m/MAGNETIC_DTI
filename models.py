import torch
from torch import nn
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.utils import expand_as_pair
import dgl.nn as dglnn
import dgl

from torch.nn import Module, ModuleList, BatchNorm1d
import torch.nn.functional as F


class AGDNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 K,
                 feat_drop=0.,
                 attn_drop=0.,
                 diffusion_drop=0.,
                 edge_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 transition_matrix='gat',
                 weight_style="HA",
                 hop_norm=False,
                 pos_emb=True,
                 bias=True,
                 share_weights=True,
                 no_dst_attn=False,
                 pre_act=False,
                 ):
        super(AGDNConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._K = K
        self._allow_zero_in_degree = allow_zero_in_degree
        self._transition_matrix = transition_matrix
        self._hop_norm = hop_norm
        self._weight_style = weight_style
        self._pos_emb = pos_emb
        self._share_weights = share_weights
        self._pre_act = pre_act
        self._edge_drop = edge_drop

        if residual:
            # if self._in_dst_feats != out_feats * num_heads:
            self.res_fc = nn.Linear(
                self._in_dst_feats, num_heads * out_feats, bias=False)
            # else:
            #     self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            if self._share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)

        if transition_matrix.startswith('gat'):
            if pre_act:
                self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            elif transition_matrix == 'gat_sym' or no_dst_attn:
                self.attn_l = self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            else:
                self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
                self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.diffusion_drop = nn.Dropout(diffusion_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if pos_emb:
            self.position_emb = nn.Parameter(torch.FloatTensor(size=(1, num_heads, K + 1, out_feats)))
        if weight_style in ["HA", "HA+HC"]:
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(num_heads,)))
        if weight_style in ["HC", "HA+HC"]:
            self.weights = nn.Parameter(torch.ones(size=(1, num_heads, K + 1, out_feats)))


        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
        self.activation = activation




        self.gat_layer = dglnn.GATConv(in_feats, out_feats, num_heads=5, attn_drop=0.1, residual=True, allow_zero_in_degree=True)


    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            # self.res_fc.reset_parameters()
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        # self.fc_src.reset_parameters()
        if not self._share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            # self.fc_dst.reset_parameters()
        if self._transition_matrix.startswith('gat'):
            if self._pre_act:
                nn.init.xavier_normal_(self.attn, gain=gain)
            else:
                nn.init.xavier_normal_(self.attn_l, gain=gain)
                nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._pos_emb:
            nn.init.xavier_normal_(self.position_emb, gain=gain)
        if self._weight_style in ["HA", "HA+HC"]:
            nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
            nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
            # nn.init.zeros_(self.hop_attn_l)
            # nn.init.zeros_(self.hop_attn_r)
            nn.init.uniform_(self.beta)

        elif self._weight_style in ["HC", "HA+HC"]:
            nn.init.xavier_uniform_(self.weights, gain=gain)
        elif self._weight_style == "lstm":
            self.lstm.reset_parameters()
            self.att.reset_parameters()
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def feat_trans(self, h, i):
        if self._hop_norm:
            h = F.normalize(h, dim=-1, p=2)
        # h = (h - h.mean(0)) / (h.std(0) + 1e-9)
        # h = (h-h.min(0)[0]) / (h.max(0)[0] - h.min(0)[0])
        if self._pos_emb:
            h = h + self.position_emb[:, :, i, :]
        # h = (0.5 ** i) * h
        return h

    def forward_old(self, graph, feat, edge_feat=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self._share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            #graph.srcdata.update({'ft': feat_src})
            graph.srcdata['ft'] = feat_src
            graph.dstdata['ft'] = feat_dst

            if self._transition_matrix.startswith('gat'):
                if self._pre_act:
                    graph.srcdata.update({'el': feat_src})
                    graph.dstdata.update({'er': feat_dst})
                else:
                    el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                    er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                    graph.srcdata.update({'el': el})
                    graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                # if edge_feat is not None:
                #     feat_e = self.fc_e(edge_feat).view(-1, self._num_heads, self._edge_feats)
                #     graph.edata['e'] = graph.edata['e'] + (feat_e * self.attn_e).sum(dim=-1).unsqueeze(-1)
                e = graph.edata.pop('e')
                e = self.leaky_relu(e)

                if self._pre_act:
                    e = (e * self.attn).sum(dim=-1).unsqueeze(-1)
                # if self.training and self._edge_drop > 0:
                #     perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                #     bound = int(graph.number_of_edges() * self._edge_drop)
                #     eids = perm[bound:]
                # else:
                #     eids = torch.arange(graph.number_of_edges(), device=graph.device)
                # compute softmax
                if self._transition_matrix == 'gat_sym':
                    a = torch.sqrt(
                        (edge_softmax(graph, e, norm_by='dst') * edge_softmax(graph, e, norm_by='src')).clamp(min=1e-9))
                elif self._transition_matrix == 'gat_col':
                    a = edge_softmax(graph, e, norm_by='src')
                else:
                    a = edge_softmax(graph, e, norm_by='dst')

                if edge_feat is not None:
                    graph.edata['w'] = edge_feat
                    graph.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'w_d'))
                    graph.apply_edges(fn.copy_u('w_d', 'w_src'))
                    graph = graph.reverse(copy_edata=True)
                    graph.apply_edges(fn.copy_u('w_d', 'w_dst'))
                    graph = graph.reverse(copy_edata=True)
                    graph.edata['w'] = graph.edata['w'] / torch.sqrt(graph.edata['w_src'] * graph.edata['w_dst'])

                    #a = (a + graph.edata['w'].unsqueeze(1)) / 2
                    a = (a + graph.edata['w'].unsqueeze(1).unsqueeze(1).expand(-1, self._num_heads, -1)) / 2

            elif self._transition_matrix == 'gcn':
                deg = graph.in_degrees()
                inv_deg = 1 / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                sqrt_inv_deg = inv_deg.pow(0.5)
                a = sqrt_inv_deg[graph.edges()[0]] * sqrt_inv_deg[graph.edges()[1]]
                a = a.view(-1, 1, 1)
            elif self._transition_matrix == 'row':
                deg = graph.in_degrees()
                inv_deg = 1. / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                a = inv_deg[graph.edges()[1]]
                a = a.view(-1, 1, 1)
            elif self._transition_matrix == 'col':
                deg = graph.out_degrees()
                inv_deg = 1. / deg
                inv_deg[torch.isinf(inv_deg)] = 0
                a = inv_deg[graph.edges()[0]]
                a = a.view(-1, 1, 1)

            # message passing

            hstack = [self.feat_trans(graph.dstdata['ft'], 0)]
            h_query = self.feat_trans(graph.dstdata['ft'], 0).unsqueeze(2)

            for k in range(1, self._K + 1):
                # Apply diffusion drop to each node type separately
                for ntype in graph.ntypes:
                    graph.nodes[ntype].data['ft'] = self.diffusion_drop(graph.nodes[ntype].data['ft'])

                graph.edata['a'] = self.attn_drop(a)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                hstack.append(self.feat_trans(graph.dstdata['ft'], k))
            hstack = torch.stack(hstack, dim=2)
            if self._weight_style in ["HC"]:
                rst = (hstack * self.attn_drop(self.weights)).sum(dim=2)
            elif self._weight_style in ["HA", "HA+HC"]:

                astack = (hstack * self.hop_attn_r.unsqueeze(2)).sum(dim=-1).unsqueeze(-1) \
                         + (h_query * self.hop_attn_l.unsqueeze(2)).sum(dim=-1).unsqueeze(-1)
                astack = self.leaky_relu(astack)
                astack = F.softmax(astack, dim=2) * torch.exp(self.beta.view(1, -1, 1, 1))
                # astack = self.attn_drop(astack)
                if self._weight_style == "HA+HC":
                    hstack = hstack * self.weights
                rst = (hstack * astack).sum(dim=2)
            elif self._weight_style == "sum":
                rst = hstack.sum(dim=2)
            elif self._weight_style == "max_pool":
                rst = hstack.max(dim=2)[0]
            elif self._weight_style == "mean_pool":
                rst = hstack.mean(dim=2)
            elif self._weight_style == "lstm":
                alpha, _ = self.lstm(hstack.view(-1, self._K + 1, self._out_feats))
                alpha = self.att(alpha)
                alpha = torch.softmax(alpha, dim=1)
                rst = (hstack * alpha.view(-1, self._num_heads, self._K + 1, 1)).sum(dim=2)

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if self._share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_src).view(
                        -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            #graph.srcdata.update({'ft': feat_src})
            graph.srcdata['ft'] = feat_src
            graph.dstdata['ft'] = feat_dst



            gat_out, a = self.gat_layer(graph, (h_src, h_dst),  get_attention=True)
            graph.dstdata['ft'] = gat_out



            # message passing
            hstack = [self.feat_trans(graph.dstdata['ft'], 0)]
            h_query = self.feat_trans(graph.dstdata['ft'], 0).unsqueeze(2)

            for k in range(1, self._K + 1):
                # Apply diffusion drop to each node type separately
                for ntype in graph.ntypes:
                    graph.nodes[ntype].data['ft'] = self.diffusion_drop(graph.nodes[ntype].data['ft'])

                graph.edata['a'] = self.attn_drop(a)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
                hstack.append(self.feat_trans(graph.dstdata['ft'], k))
            hstack = torch.stack(hstack, dim=2)

            if self._weight_style in ["HA", "HA+HC"]:

                astack = (hstack * self.hop_attn_r.unsqueeze(2)).sum(dim=-1).unsqueeze(-1) \
                         + (h_query * self.hop_attn_l.unsqueeze(2)).sum(dim=-1).unsqueeze(-1)
                astack = self.leaky_relu(astack)
                astack = F.softmax(astack, dim=2) * torch.exp(self.beta.view(1, -1, 1, 1))
                # astack = self.attn_drop(astack)
                if self._weight_style == "HA+HC":
                    hstack = hstack * self.weights
                rst = (hstack * astack).sum(dim=2)


            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
