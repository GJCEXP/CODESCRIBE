# coding=utf-8
import os
import re
import sys
from my_lib.neural_module.learn_strategy import LrWarmUp
from my_lib.neural_module.transformer import TranEnc, TranDec, DualTranDec
from my_lib.neural_module.embedding import PosEnc
from my_lib.neural_module.loss import LabelSmoothSoftmaxCEV2, CriterionNet
from my_lib.neural_module.balanced_data_parallel import BalancedDataParallel
from my_lib.neural_module.copy_attention import DualMultiCopyGenerator
from my_lib.neural_module.beam_search import trans_beam_search
from my_lib.neural_model.seq_to_seq_model import TransSeq2Seq
from my_lib.neural_model.base_model import BaseNet

from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataListLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import DataParallel
import random
import numpy as np
import os
import logging
import pickle
import json
import codecs
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Datax(Data):
    def __init__(self,
                 ast_node=None, ast_edge=None, ast_pos=None,ast_src_map=None,
                 code_token=None, code_src_map=None,
                 text_token_input=None, text_token_output=None,
                 idx=None):
        super().__init__()
        self.__num_nodes__=ast_node.size(0)   #DataParallel的时候要用

        self.ast_node = ast_node  # (num_ast_node,) dim_cat=0
        self.ast_edge = ast_edge  # (2,num_ast_edge) dim_cat=-1
        self.ast_pos = ast_pos  # (num_ast_node,) dim_cat=0
        self.ast_src_map=ast_src_map  # (num_ast_node,) dim_cat=0
        self.code_token = code_token  # (num_diff_token_before,) dim_cat=None New0
        self.code_src_map = code_src_map  # (num_diff_token_before,) dim_cat=None New0
        self.text_token_input = text_token_input  # (num_token_text,) dim_cat=None New0  decoder的输入端
        self.text_token_output = text_token_output  # (num_token_text,) dim_cat=None New0    decoder的输出端
        self.idx = idx.expand(1) if idx is not None else None  # (1,) dim_cat=0

    def __inc__(self, key, value):  # 增量，针对edge_index
        if key == 'ast_edge':
            return self.ast_node.size(0)
        else:
            return 0

    def __cat_dim__(self, key, value):
        if bool(re.search('edge', key)):
            return -1
        elif bool(re.search('(node|pos)', key)) or key == 'idx':
            return 0
        elif bool(re.search('(token|map)', key)):
            return None  # generate a new 0 dimension
        else:
            super().__cat_dim__(key, value)

class Datasetx(Dataset):
    '''
    文本对数据集对象（根据具体数据再修改）
    '''

    def __init__(self,
                 code_asts,
                 texts=None,
                 ids=None,
                 code_max_len=None,
                 text_max_len=None,
                 ast_max_size=None,
                 text_begin_idx=1,
                 text_end_idx=2,
                 pad_idx=0):
        self.len = len(code_asts)  # 样本个数
        self.code_max_len = code_max_len
        self.ast_max_size=ast_max_size
        self.text_max_len = text_max_len
        self.text_begin_idx = text_begin_idx
        self.text_end_idx = text_end_idx
        
        if code_max_len is None:
            self.code_max_len = max([len(item['code']['tokens']) for item in code_asts])
        if ast_max_size is None:
            self.ast_max_size=max(len(item['ast']['nodes']) for item in code_asts)
        if text_max_len is None and texts is not None:
            self.text_max_len = max([len(text) for text in texts])  # 每个输出只是一个序列
        self.code_asts = code_asts
        self.texts = texts
        self.ids = ids
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        tru_code_tokens = self.code_asts[index]['code']['tokens'][:self.code_max_len]
        pad_code_tokens = np.lib.pad(tru_code_tokens,
                                            (0, self.code_max_len - len(tru_code_tokens)),
                                            'constant',
                                            constant_values=(self.pad_idx, self.pad_idx))
        tru_code_src_map = self.code_asts[index]['code']['src_map'][:self.code_max_len]
        pad_code_src_map = np.lib.pad(tru_code_src_map,
                                             (0, self.code_max_len - len(tru_code_tokens)),
                                             'constant',
                                             constant_values=(self.pad_idx, self.pad_idx))
        tru_ast_src_map = self.code_asts[index]['ast']['src_map'][:self.ast_max_size]
        pad_ast_src_map = np.lib.pad(tru_ast_src_map,
                                  (0, self.ast_max_size - len(tru_ast_src_map)),
                                  'constant',
                                  constant_values=(self.pad_idx, self.pad_idx))
        if self.texts is None:
            pad_text_in = np.zeros((self.text_max_len + 1,), dtype=np.int64)  # decoder端的输入
            pad_text_in[0] = self.text_begin_idx
            pad_text_out = None
        else:
            tru_text = self.texts[index][:self.text_max_len]  # 先做截断
            pad_text_in = np.lib.pad(tru_text,
                                    (1, self.text_max_len - len(tru_text)),
                                    'constant',
                                    constant_values=(self.text_begin_idx, self.pad_idx))
            tru_text_out = np.lib.pad(tru_text,
                                     (0, 1),
                                     'constant',
                                     constant_values=(0, self.text_end_idx))  # padding
            pad_text_out = np.lib.pad(tru_text_out,
                                     (0, self.text_max_len + 1 - len(tru_text_out)),
                                     'constant',
                                     constant_values=(self.pad_idx, self.pad_idx))  # padding
            # pad_out_input=np.lib.pad(pad_out[:-1],(1,0),'constant',constant_values=(self.text_begin_idx, 0))
        return Datax(ast_node=torch.tensor(self.code_asts[index]['ast']['nodes']),
                     ast_edge=torch.tensor(self.code_asts[index]['ast']['edges'], dtype=torch.long),
                     ast_pos=torch.tensor(self.code_asts[index]['ast']['poses']),
                     ast_src_map=torch.tensor(pad_ast_src_map).long(),
                     code_token=torch.tensor(pad_code_tokens).long(),
                     code_src_map=torch.tensor(pad_code_src_map).long(),
                     text_token_input=torch.tensor(pad_text_in).long(),
                     text_token_output=torch.tensor(pad_text_out).long() if self.texts is not None else None,
                     idx=torch.tensor(self.ids[index]) if self.ids is not None else None)  # 防止没有id

    def __len__(self):
        return self.len


# GNN=SAGEConv
class ASTEnc(nn.Module):
    def __init__(self,
                 # node_voc_size,
                 pos_voc_size,
                 emb_dims,
                 hid_dims,
                 gnn_layers,
                 GNN=SAGEConv,
                 aggr='add',
                 drop_rate=0.,
                 **kwargs):
        '''
        AST编码器
        :param node_voc_size: the size of node vocabulary   
        :param pos_voc_size:  the size of position vocabulary
        :param emb_dims: the dims of embebddings
        :param hid_dims: the dims of hidden layer
        :param gnn_layers: the number of layers
        :param GNN: GNN module
        :param aggr: the mode of aggregation
        :param drop_rate: the dropout rate
        :param kwargs: others
        '''
        super().__init__()
        # kwargs.setdefault('init_emb','None')
        # kwargs.setdefault('batch','None')   #GraphData.batch to_dense_data用的,放到forward里了
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        kwargs.setdefault('ast_max_size', None)  # GraphData.batch to_dense_data用的
        self.emb_dims = emb_dims
        self.pad_idx = kwargs['pad_idx']
        # self.batch=kwargs['batch']
        self.ast_max_size = kwargs['ast_max_size']
        # self.node_embedding = nn.Embedding(node_voc_size, emb_dims, padding_idx=self.pad_idx)  # node embedding层
        # nn.init.xavier_uniform_(self.node_embedding.weight[1:,])
        self.pos_embedding = nn.Embedding(pos_voc_size, emb_dims, padding_idx=self.pad_idx)
        nn.init.xavier_uniform_(self.pos_embedding.weight[1:, ])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        self.gnn_layers = gnn_layers
        gnn1=GNN(in_channels=emb_dims, out_channels=hid_dims, aggr=aggr)
        gnn2=GNN(in_channels=hid_dims, out_channels=emb_dims, aggr=aggr)
        if gnn_layers==2:
            self.gnns=nn.ModuleList([gnn1,gnn2])
            self.hid_layer_norms=nn.ModuleList([nn.LayerNorm(hid_dims),nn.LayerNorm(emb_dims),])
        if self.gnn_layers>2:
            self.gnns=nn.ModuleList([gnn1]+[GNN(in_channels=hid_dims, out_channels=hid_dims, aggr=aggr)
                                            for _ in range(gnn_layers-2)]+[gnn2])
            self.hid_layer_norms=nn.ModuleList([nn.LayerNorm(hid_dims) for _ in range(gnn_layers-1)]+[nn.LayerNorm(emb_dims)])
        self.relus=nn.ModuleList([nn.Sequential(nn.ReLU(),nn.Dropout(p=drop_rate)) for _ in range(gnn_layers)])
        # self.linear=nn.Linear(hid_dims, emb_dims)
        self.out_layer_norm=nn.LayerNorm(emb_dims)
        # self.gnns = nn.ModuleList(
        #     [GNN(in_channels=emb_dims, out_channels=hid_dims, aggr=aggr) for _ in range(gnn_layers)])
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(emb_dims) for _ in range(gnn_layers)])
        # self.linears = nn.ModuleList([nn.Linear(hid_dims, emb_dims) for _ in range(gnn_layers)])
        self.dropout = nn.Dropout(p=drop_rate)
        # self.layer_norm=nn.LayerNorm(out_dims, elementwise_affine=True)

    def forward(self, node_emb, pos, edge, node_batch=None):
        '''
        :param node:
        :param pos:
        :param edge:
        :param batch: [batch_node_num,]
        :return:
        '''
        assert len(node_emb.size()) == 2  # node是一堆节点序号[batch_node_num,emb_dims]
        assert len(pos.size()) == 1  # pos是一堆节点序号[batch_node_num,]
        assert len(edge.size()) == 2  # 点是一堆节点序号[2,all_batch_edge_num]
        node_emb = node_emb * np.sqrt(self.emb_dims)  # [batch_node_num,emb_dims]
        pos_emb = self.pos_embedding(pos)  # [batch_node_num,emb_dims]
        node_enc = self.dropout(node_emb.add(pos_emb))  # [batch_node_num,emb_dims]
        node_enc = self.emb_layer_norm(node_enc)  # [batch_node_num,emb_dims]

        for i in range(self.gnn_layers):
            node_enc_=self.gnns[i](x=node_enc,edge_index=edge)   # [batch_node_num,hid_dims]
            node_enc_=self.relus[i](node_enc_)    # [batch_node_num,hid_dims]
            node_enc=self.hid_layer_norms[i](node_enc_.add(node_enc))  # [batch_node_num,hid_dims]
        # node_enc=self.linear(node_enc)
        if node_batch is not None:
            node_enc = to_dense_batch(node_enc,
                                      batch=node_batch,
                                      fill_value=self.pad_idx,
                                      max_num_nodes=self.ast_max_size)[0]  # [batch_ast_num,ast_max_size,emb_dims],必须要用[0]，不然是个tuple
            # node_enc=self.out_layer_norm(node_enc)
        return node_enc  # [batch_node_num,emb_dims] 或者 #[batch_ast_num,ast_max_size,emb_dims]

class CodeEnc(nn.Module):
    def __init__(self,
                 max_len,
                 # voc_size,
                 emb_dims,
                 att_layers=3,
                 att_heads=8,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        self.emb_dims = emb_dims
        # self.token_embedding = nn.Embedding(voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        # nn.init.xavier_uniform_(self.token_embedding.weight[1:, ])
        self.pos_encoding = PosEnc(max_len=max_len, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        # self.code_enc = TranDec(query_dims=emb_dims,
        #                         key_dims=ast_enc_out_dims,
        #                         head_num=att_heads,
        #                         ff_hid_dims=ff_hid_dims,
        #                         head_dims=att_head_dims,
        #                         layer_num=att_layers,
        #                         drop_rate=drop_rate,
        #                         pad_idx=kwargs['pad_idx'],
        #                         self_causality=False)
        self.code_enc = TranEnc(query_dims=emb_dims,
                                head_num=att_heads,
                                ff_hid_dims=ff_hid_dims,
                                head_dims=att_head_dims,
                                layer_num=att_layers,
                                drop_rate=drop_rate,
                                pad_idx=kwargs['pad_idx'])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, code_emb):
        '''
        :param diff:   [batch_data_num,code_max_len]
        :param node:    [batch_data_num,ast_max_size,node_emb_dims]
        :return:
        '''
        code_emb = code_emb * np.sqrt(self.emb_dims)  # [batch_data_num,code_max_len,code_emb_dims]
        pos_emb = self.pos_encoding(code_emb)  # [batch_data_num,code_max_len,code_emb_dims]
        code_enc = self.dropout(code_emb.add(pos_emb))  # [batch_data_num,code_max_len,code_emb_dims]
        # code_enc=self.dropout(code_emb)
        code_enc = self.emb_layer_norm(code_enc)  # [batch_data_num,code_max_len,code_emb_dims]
        diff_mask = code_emb.abs().sum(-1).sign()  # [batch_data_num,code_max_len]

        code_enc = self.code_enc(query=code_enc,
                                 query_mask=diff_mask)  # [batch_data_num,code_max_len,code_emb_dims]

        return code_enc

class ASTCodeEnc(nn.Module):
    def __init__(self,
                 ast_emb_dims,
                 code_emb_dims,
                 ast_max_size,
                 code_max_len,
                 code_node_voc_size,
                 ast_pos_voc_size,
                 # code_voc_size,
                 ast_hid_dims,
                 ast_gnn_layers,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='mean',
                 code_att_layers=3,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.code_node_embedding = nn.Embedding(code_node_voc_size, emb_dims, padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(self.code_node_embedding.weight[1:, ])
        self.ast_enc = ASTEnc(pos_voc_size=ast_pos_voc_size,
                              emb_dims=ast_emb_dims,
                              hid_dims=ast_hid_dims,
                              gnn_layers=ast_gnn_layers,
                              GNN=ast_GNN,
                              aggr=ast_gnn_aggr,
                              drop_rate=drop_rate,
                              ast_max_size=ast_max_size,
                              pad_idx=kwargs['pad_idx'])
        self.code_enc = CodeEnc(max_len=code_max_len,
                                emb_dims=code_emb_dims,
                                att_layers=code_att_layers,
                                att_heads=code_att_heads,
                                att_head_dims=code_att_head_dims,
                                ff_hid_dims=code_ff_hid_dims,
                                drop_rate=drop_rate,
                                pad_idx=kwargs['pad_idx'])

    def forward(self, data):
        node_emb=self.code_node_embedding(data.ast_node)
        code_emb=self.code_node_embedding(data.code_token)
        ast_enc_out = self.ast_enc(node_emb=node_emb, pos=data.ast_pos, edge=data.ast_edge, node_batch=data.ast_node_batch)
        code_enc_out = self.code_enc(code_emb=code_emb)
        return ast_enc_out,code_enc_out

class Dec(nn.Module):
    def __init__(self,
                 emb_dims,
                 text_voc_size,
                 text_max_len,
                 enc_max_len,
                 enc_out_dims,
                 att_layers,
                 att_heads,
                 att_head_dims=None,
                 ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs
                 ):
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        kwargs.setdefault('copy', True)
        self._copy = kwargs['copy']
        # self.text_voc_size=text_voc_size
        self.emb_dims = emb_dims
        # self.text_max_len = text_max_len
        self.text_voc_size = text_voc_size
        # embedding dims为text_voc_size+2*text_max_len
        self.text_embedding = nn.Embedding(text_voc_size + enc_max_len, emb_dims, padding_idx=kwargs['pad_idx'])
        nn.init.xavier_uniform_(self.text_embedding.weight[1:, ])
        self.pos_encoding = PosEnc(max_len=text_max_len+1, emb_dims=emb_dims, train=True, pad=True,pad_idx=kwargs['pad_idx'])  #不要忘了+1,因为输入前加了begin_id
        # nn.init.xavier_uniform_(self.pos_encoding.weight[1:, ])
        self.emb_layer_norm = nn.LayerNorm(emb_dims)
        # self.text_dec = TranDec(query_dims=emb_dims,
        #                         key_dims=enc_out_dims,
        #                         head_num=att_heads,
        #                         ff_hid_dims=ff_hid_dims,
        #                         head_dims=att_head_dims,
        #                         layer_num=att_layers,
        #                         drop_rate=drop_rate,
        #                         pad_idx=kwargs['pad_idx'],
        #                         self_causality=True)
        self.text_dec = DualTranDec(query_dims=emb_dims,
                                   key_dims=enc_out_dims,
                                   head_nums=att_heads,
                                   head_dims=att_head_dims,
                                   layer_num=att_layers,
                                   ff_hid_dims=ff_hid_dims,
                                   drop_rate=drop_rate,
                                   pad_idx=kwargs['pad_idx'],
                                   mode='sequential')
        self.dropout = nn.Dropout(p=drop_rate)
        self.out_fc = nn.Linear(emb_dims, text_voc_size)
        self.copy_generator = DualMultiCopyGenerator(tgt_dims=emb_dims,
                                                     tgt_voc_size=text_voc_size,
                                                     src_dims=enc_out_dims,
                                                     att_heads=att_heads,
                                                     att_head_dims=att_head_dims,
                                                     drop_rate=drop_rate,
                                                     pad_idx=kwargs['pad_idx'])

    def forward(self,
                ast_enc_out,code_enc_out,
                ast_src_map,code_src_map,
                text_input):
        text_emb = self.text_embedding(text_input) * np.sqrt(self.emb_dims)  # (B,L_text,D_text_emb)
        pos_emb = self.pos_encoding(text_input)  # # (B,L_text,D_emb)
        text_dec = self.dropout(text_emb.add(pos_emb))  # (B,L_text,D_emb)
        text_dec = self.emb_layer_norm(text_dec)  # (B,L_text,D_emb)

        # enc_out=torch.cat([ast_enc_out,code_enc_out,],dim=1)
        # enc_map=torch.cat([ast_src_map,code_src_map,],dim=1)

        # enc_mask = enc_out.abs().sum(-1).sign()  # (B,L_diff)
        # text_mask = text_input.abs().sign()  # (B,L_text)
        # text_dec = self.text_dec(query=text_dec,
        #                        key=enc_out,
        #                        query_mask=text_mask,
        #                        key_mask=enc_mask
        #                        )  # (B,L_text,D_text_emb)

        # ast_mask = ast_enc_out.abs().sum(-1).sign()  # (B,L_diff)
        ast_mask = ast_enc_out.abs().sum(-1).sign()  # (B,L_diff)
        code_mask = code_enc_out.abs().sum(-1).sign()  # (B,L_diff)
        text_mask = text_input.abs().sign()  # (B,L_text)
        text_dec = self.text_dec(query=text_dec,
                                 key1=ast_enc_out,
                                 key2=code_enc_out,
                                 query_mask=text_mask,
                                 key_mask1=ast_mask,
                                 key_mask2=code_mask,
                                 )  # (B,L_text,D_text_emb)

        if not self._copy:
            text_output = self.out_fc(text_dec)  # (B,L_text,text_voc_size)包含begin_idx和end_idx
            # text_output = F.softmax(text_output, dim=-1)
            # text_output[:,:,-1]=0.    #不生成begin_idx，默认该位在text_voc_size最后一个，置0
        else:
            # text_output=F.pad(text_output,(0,2*self.text_max_len)) #pad last dim
            text_output = self.copy_generator(text_dec,
                                             code_enc_out,code_src_map,
                                              ast_enc_out,ast_src_map)
        # text_output[:, :, self.text_voc_size - 1] = 0.  # 不生成begin_idx，默认该位在text_voc_size最后一个，置0
        # text_output[:, :, 0] = 0.  # pad位不生成
        return text_output.transpose(1, 2)

class TNet(BaseNet):
    def __init__(self,
                 ast_emb_dims,
                 code_emb_dims,
                 text_emb_dims,
                 ast_max_size,
                 code_max_len,
                 text_max_len,
                 code_node_voc_size,
                 ast_pos_voc_size,
                 # code_voc_size,
                 text_voc_size,
                 ast_hid_dims,
                 ast_gnn_layers,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='add',
                 code_att_layers=3,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 **kwargs,
                 ):
        super().__init__()
        kwargs.setdefault('copy', True)
        kwargs.setdefault('pad_idx', 0)  # GraphData.batch to_dense_data用的
        self.init_params = locals()
        self.enc = ASTCodeEnc(ast_emb_dims=ast_emb_dims,
                              code_emb_dims=code_emb_dims,
                              ast_max_size=ast_max_size,
                              code_max_len=code_max_len,
                              code_node_voc_size=code_node_voc_size,
                              ast_pos_voc_size=ast_pos_voc_size,
                              # code_voc_size=code_node_voc_size,
                              ast_hid_dims=ast_hid_dims,
                              ast_gnn_layers=ast_gnn_layers,
                              ast_GNN=ast_GNN,
                              ast_gnn_aggr=ast_gnn_aggr,
                              code_att_layers=code_att_layers,
                              code_att_heads=code_att_heads,
                              code_att_head_dims=code_att_head_dims,
                              code_ff_hid_dims=code_ff_hid_dims,
                              drop_rate=drop_rate,
                              pad_idx=kwargs['pad_idx'])
        self.dec = Dec(emb_dims=text_emb_dims,
                       text_voc_size=text_voc_size,
                       text_max_len=text_max_len,
                       enc_max_len=code_max_len+ast_max_size,
                       enc_out_dims=ast_emb_dims,
                       att_layers=text_att_layers,
                       att_heads=text_att_heads,
                       att_head_dims=text_att_head_dims,
                       ff_hid_dims=text_ff_hid_dims,
                       drop_rate=drop_rate,
                       copy=kwargs['copy'],
                       pad_idx=kwargs['pad_idx'])

    def forward(self, code_ast):
        ast_enc_out, code_enc_out = self.enc(data=code_ast)
        text_output = self.dec(ast_enc_out=ast_enc_out,
                              ast_src_map=code_ast.ast_src_map,
                              code_enc_out=code_enc_out,
                              code_src_map=code_ast.code_src_map,
                              text_input=code_ast.text_token_input)
        return text_output

class TModel(TransSeq2Seq):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 ast_emb_dims=512,
                 code_emb_dims=512,
                 text_emb_dims=512,
                 ast_hid_dims=1028,
                 ast_gnn_layers=3,
                 ast_GNN=SAGEConv,
                 ast_gnn_aggr='add',
                 code_att_layers=3,
                 code_att_heads=8,
                 code_att_head_dims=None,
                 code_ff_hid_dims=2048,
                 text_att_layers=3,
                 text_att_heads=8,
                 text_att_head_dims=None,
                 text_ff_hid_dims=2048,
                 drop_rate=0.,
                 copy=True,
                 pad_idx=0,
                 train_batch_size=32,
                 pred_batch_size=32,
                 max_big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 start_valid_epoch=20,
                 early_stop=20,
                 Net=TNet,
                 Dataset=Datasetx,
                 beam_width=1,
                 train_metrics=[get_sent_bleu],
                 valid_metric=get_sent_bleu,
                 test_metrics=[get_sent_bleu],
                 train_mode=True,
                 **kwargs
                 ):
        logging.info('Construct %s' % model_name)
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)
        self.init_params = locals()
        self.ast_emb_dims = ast_emb_dims
        self.code_emb_dims = code_emb_dims
        self.text_emb_dims = text_emb_dims
        self.ast_hid_dims = ast_hid_dims
        self.ast_gnn_layers = ast_gnn_layers
        self.ast_GNN = ast_GNN
        self.ast_gnn_aggr = ast_gnn_aggr
        self.code_att_layers = code_att_layers
        self.code_att_heads = code_att_heads
        self.code_att_head_dims = code_att_head_dims
        self.code_ff_hid_dims = code_ff_hid_dims
        self.text_att_layers = text_att_layers
        self.text_att_heads = text_att_heads
        self.text_att_head_dims = text_att_head_dims
        self.text_ff_hid_dims = text_ff_hid_dims
        self.drop_rate = drop_rate
        self.pad_idx = pad_idx
        self.copy = copy
        self.train_batch_size = train_batch_size
        self.pred_batch_size = pred_batch_size
        self.max_big_epochs = max_big_epochs
        self.regular_rate = regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate = min_lr_rate
        self.warm_big_epochs = warm_big_epochs
        self.start_valid_epoch=start_valid_epoch
        self.early_stop=early_stop
        self.Net = Net
        self.Dataset = Dataset
        self.beam_width = beam_width
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        self.test_metrics = test_metrics
        self.train_mode = train_mode

    def fit(self,
            train_data,
            valid_data,
            **kwargs
            ):
        self.ast_max_size = 0
        self.code_max_len = 0
        self.code_node_voc_size = 0
        self.ast_pos_voc_size = 0
        # self.code_voc_size = 0
        for item in train_data['code_asts']:
            self.ast_max_size = max(self.ast_max_size,len(item['ast']['nodes']))
            self.code_max_len = max(self.code_max_len,len(item['code']['tokens']))
            # self.code_max_len = 200
            self.code_node_voc_size = max(self.code_node_voc_size,
                                         max(item['ast']['nodes'] + item['code']['tokens']))
            self.ast_pos_voc_size = max(self.ast_pos_voc_size,max(item['ast']['poses']))
        self.code_node_voc_size+=1
        self.ast_pos_voc_size+=1
        self.text_max_len = max([len(text) for text in train_data['texts']])
        # self.text_max_len = 30
        self.text_voc_size = len(train_data['text_dic']['text_i2w'])  # 包含了begin_idx和end_idx
        print(self.code_max_len,self.ast_max_size, self.text_max_len,
              self.code_node_voc_size, self.ast_pos_voc_size,self.text_voc_size)

        net = self.Net(ast_emb_dims=self.ast_emb_dims,
                       code_emb_dims=self.code_emb_dims,
                       text_emb_dims=self.text_emb_dims,
                       ast_max_size=self.ast_max_size,
                       code_max_len=self.code_max_len,
                       text_max_len=self.text_max_len,
                       code_node_voc_size=self.code_node_voc_size,
                       ast_pos_voc_size=self.ast_pos_voc_size,
                       # code_voc_size=self.code_voc_size,
                       text_voc_size=self.text_voc_size,
                       ast_hid_dims=self.ast_hid_dims,
                       ast_gnn_layers=self.ast_gnn_layers,
                       ast_GNN=self.ast_GNN,
                       ast_gnn_aggr=self.ast_gnn_aggr,
                       code_att_layers=self.code_att_layers,
                       code_att_heads=self.code_att_heads,
                       code_att_head_dims=self.code_att_head_dims,
                       code_ff_hid_dims=self.code_ff_hid_dims,
                       text_att_layers=self.text_att_layers,
                       text_att_heads=self.text_att_heads,
                       text_att_head_dims=self.text_att_head_dims,
                       text_ff_hid_dims=self.text_ff_hid_dims,
                       drop_rate=self.drop_rate,
                       pad_idx=self.pad_idx,
                       copy=self.copy
                       )
        logging.info("{} have {} paramerters in total".format(self.model_name, sum(x.numel() for x in net.parameters() if x.requires_grad)))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先

        self.net =DataParallel(net.to(device),follow_batch=['ast_node'])  # 并行使用多GPU
        # self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
        # self.net = net.to(device)  # 数据转移到设备

        self.net.train()  # 设置网络为训练模式

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr_base,
                                    weight_decay=self.regular_rate)

        # ast_enc_params=self.net.module.enc.ast_enc.parameters()
        # ast_enc_param_ids=list(map(id,ast_enc_params))
        # ex_params=filter(lambda p: id(p) not in ast_enc_param_ids,self.net.parameters())
        # optim_cfg = [{'params': ast_enc_params, 'lr': 0.001,'weight_decay': self.regular_rate* 10.},
        #              {'params': ex_params, 'lr': self.lr_base, 'weight_decay': self.regular_rate}]
        # self.optimizer=optim.Adam(optim_cfg)

        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean', ignore_index=self.pad_idx, label_smooth=0.0)
        # self.criterion = nn.NLLLoss(ignore_index=self.pad_idx)

        self.text_begin_idx = self.text_voc_size - 1
        self.text_end_idx = self.text_voc_size - 2
        self.tgt_begin_idx,self.tgt_end_idx=self.text_begin_idx,self.text_end_idx
        assert train_data['text_dic']['text_i2w'][self.text_end_idx] == OUT_END_TOKEN
        assert train_data['text_dic']['text_i2w'][self.text_begin_idx] == OUT_BEGIN_TOKEN  # 最后两个是end_idx 和begin_idx

        train_set = self.Dataset(code_asts=train_data['code_asts'],
                                 texts=train_data['texts'],
                                 ids=train_data['ids'],
                                 code_max_len=self.code_max_len,
                                 text_max_len=self.text_max_len,
                                 ast_max_size=self.ast_max_size,
                                 text_begin_idx=self.text_begin_idx,
                                 text_end_idx=self.text_end_idx,
                                 pad_idx=self.pad_idx)
        # train_loader = DataLoader(dataset=train_set,
        #                           train_batch_size=self.train_batch_size,
        #                           shuffle=True,
        #                           follow_batch=['ast_node', 'ast_node_after'])
        train_loader=DataListLoader(dataset=train_set,
                                    batch_size=self.train_batch_size,
                                    shuffle=True,
                                    drop_last=False,)

        if self.warm_big_epochs is None:
            self.warm_big_epochs = max(self.max_big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                                  min_rate=self.min_lr_rate,
                                  lr_decay=self.lr_decay,
                                  warm_steps=self.warm_big_epochs * len(train_loader),
                                  # max(self.max_big_epochs//10,2)*train_loader.__len__()
                                  reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()
        if self.train_mode:  # 如果进行训练
            # best_net_path = os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
            # self.net.load_state_dict(torch.load(best_net_path))
            # self.net.train()
            for i in range(0,self.max_big_epochs):
                # logging.info('---------Train big epoch %d/%d' % (i + 1, self.max_big_epochs))
                pbar = tqdm(train_loader)
                for j, batch_data in enumerate(pbar):
                    batch_text_output = []
                    ids=[]
                    for data in batch_data:
                        batch_text_output.append(data.text_token_output.unsqueeze(0))
                        ids.append(data.idx.item())
                        data.text_token_output = None
                        data.idx=None

                    batch_text_output = torch.cat(batch_text_output, dim=0).to(device)
                    # print(batch_text_output[:2,:])
                    pred_text_output = self.net(batch_data)

                    loss = self.criterion(pred_text_output, batch_text_output)  # 计算loss
                    self.optimizer.zero_grad()  # 梯度置0
                    loss.backward()  # 反向传播
                    # clip_grad_norm_(self.net.parameters(),1e-2)  #减弱梯度爆炸
                    self.optimizer.step()  # 优化
                    self.scheduler.step()  # 衰减

                    log_info = '[Big epoch:{}/{}]'.format(i + 1, self.max_big_epochs)
                    if i+1>=self.start_valid_epoch:
                        text_dic = {'text_i2w': train_data['text_dic']['text_i2w'],
                                   'ex_text_i2ws': [train_data['text_dic']['ex_text_i2ws'][k] for k in ids]}
                        log_info=self._get_log_fit_eval(loss=loss,
                                                        pred_tgt=pred_text_output,
                                                        gold_tgt=batch_text_output,
                                                        tgt_i2w=text_dic
                                                        )
                        log_info = '[Big epoch:{}/{},{}]'.format(i + 1, self.max_big_epochs, log_info)
                    pbar.set_description(log_info)
                    del pred_text_output,batch_text_output,batch_data

                del pbar
                if i+1 >= self.start_valid_epoch:
                    worse_epochs=self._do_validation(valid_srcs=valid_data['code_asts'],
                                                     valid_tgts=valid_data['texts'],
                                                     tgt_i2w=valid_data['text_dic'],
                                                     increase_better=True,
                                                     last=False)  # 根据验证集loss选择best_net
                    if worse_epochs>=self.early_stop:
                        break

        self._do_validation(valid_srcs=valid_data['code_asts'],
                            valid_tgts=valid_data['texts'],
                            tgt_i2w=valid_data['text_dic'],
                            increase_better=True,
                            last=True)  # 根据验证集loss选择best_net

    def predict(self,
                code_asts,
                text_dic):
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
        self.net.eval()  # 切换测试模式
        enc=DataParallel(self.net.module.enc,follow_batch=['ast_node', 'ast_node_after'])
        dec=torch.nn.DataParallel(self.net.module.dec)
        # dec= BalancedDataParallel(0, self.net.module.dec.to(device), dim=0)  # 并行使用多GPU
        data_set = self.Dataset(code_asts=code_asts,
                                texts=None,
                                ids=None,
                                code_max_len=self.code_max_len,
                                text_max_len=self.text_max_len,
                                ast_max_size=self.ast_max_size,
                                text_begin_idx=self.text_begin_idx,
                                text_end_idx=self.text_end_idx,
                                pad_idx=self.pad_idx)  # 数据集，没有out，不需要id

        data_loader = DataListLoader(dataset=data_set,
                                     batch_size=self.pred_batch_size,   #1.5,2.5
                                     shuffle=False)
                                 # follow_batch=['ast_node', 'ast_node_after'])  # data loader
        pred_text_id_np_batches = []  # 所有batch的预测出的id np
        with torch.no_grad():  # 取消梯度
            pbar = tqdm(data_loader)
            for batch_data in pbar:
                # 从batch_data图里把解码器输入输出端数据调出来
                # text_input = batch_data.text_token_input.to(device)
                # text_output = batch_data.text_token_output.to(device)
                batch_text_input = []
                batch_code_src_map=[]
                batch_ast_src_map=[]
                for data in batch_data:
                    batch_text_input.append(data.text_token_input.unsqueeze(0))
                    batch_ast_src_map.append(data.ast_src_map.unsqueeze(0))
                    batch_code_src_map.append(data.code_src_map.unsqueeze(0))
                    data.text_token_input = None
                    data.code_src_map = None
                    data.ast_src_map=None
                batch_text_input = torch.cat(batch_text_input, dim=0).to(device)
                batch_ast_src_map = torch.cat(batch_ast_src_map, dim=0).to(device)
                batch_code_src_map = torch.cat(batch_code_src_map, dim=0).to(device)

                # 先跑encoder，生成编码
                batch_ast_enc_out, batch_code_enc_out =enc(batch_data)

                batch_text_output: list = []  # 每步的output tensor
                if self.beam_width == 1:
                    for i in range(self.text_max_len + 1):  # 每步开启
                        pred_out = dec(ast_enc_out=batch_ast_enc_out,
                                       code_enc_out=batch_code_enc_out,
                                       ast_src_map=batch_ast_src_map,
                                       code_src_map=batch_code_src_map,
                                       text_input=batch_text_input)  # 预测该步输出 (B,text_voc_size,L_text)
                        batch_text_output.append(pred_out[:, :, i].unsqueeze(-1).to('cpu').data.numpy())  # 将该步输出加入msg output
                        if i < self.text_max_len:  # 如果没到最后，将id加入input
                            batch_text_input[:, i + 1] = torch.argmax(pred_out[:, :, i], dim=1)
                    batch_pred_text = np.concatenate(batch_text_output, axis=-1)[:, :, :-1]  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.tgt_begin_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text[:, self.pad_idx, :] = -np.inf  # (B,D_tgt,L_tgt)
                    batch_pred_text_np = np.argmax(batch_pred_text, axis=1)  # (B,L_tgt) 要除去pad id和begin id
                    pred_text_id_np_batches.append(batch_pred_text_np)  # [(B,L_tgt)]
                else:
                    batch_pred_text=trans_beam_search(net=dec,
                                                      beam_width=self.beam_width,
                                                      dec_input_arg_name='text_input',
                                                      length_penalty=1,
                                                      begin_idx=self.tgt_begin_idx,
                                                      pad_idx=self.pad_idx,
                                                      end_idx=self.tgt_end_idx,
                                                      ast_enc_out=batch_ast_enc_out,
                                                      code_enc_out=batch_code_enc_out,
                                                      ast_src_map=batch_ast_src_map,
                                                      code_src_map=batch_code_src_map,
                                                      text_input=batch_text_input
                                                      )     # (B,L_tgt)

                    pred_text_id_np_batches.append(batch_pred_text.to('cpu').data.numpy()[:,:-1])  # [(B,L_tgt)]

        pred_text_id_np = np.concatenate(pred_text_id_np_batches,axis=0)  # (AB,tgt_voc_size,L_tgy)
        self.net.train()  # 切换回训练模式
        # pred_texts=[[{**text_dic['text_i2w'],**text_dic['ex_text_i2ws'][j]}[i] for ]]
        # 利用字典将msg转为token
        pred_texts = self._tgt_ids2tokens(pred_text_id_np, text_dic, self.text_end_idx)

        return pred_texts  # 序列概率输出形状为（A,D)
    
    def generate_texts(self,code_asts,text_dic,res_path,code_i2w,gold_texts,token_data,**kwargs):
        '''
        生成src对应的tgt并保存
        :param code_asts:
        :param text_dic:
        :param res_path:
        :param kwargs:
        :return:
        '''
        logging.info('>>>>>>>Generate the targets according to sources and save the result to {}'.format(res_path))
        kwargs.setdefault('beam_width',1)
        res_dir=os.path.dirname(res_path)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        pred_texts=self.predict(code_asts=code_asts,
                                text_dic=text_dic
                                )
        codes=map(lambda x:x['code']['tokens'],code_asts)
        codes=self._code_ids2tokens(codes,code_i2w,self.pad_idx)
        gold_texts=self._tgt_ids2tokens(gold_texts,text_dic,self.pad_idx)
        res_data = []
        for i,(pred_text,gold_text,code,token_item) in \
                enumerate(zip(pred_texts,gold_texts,codes,token_data)):
            sent_bleu=self.valid_metric([pred_text],[gold_text])
            res_data.append(dict(pred_text=' '.join(pred_text),
                                 gold_text=' '.join(gold_text),
                                 sent_bleu=sent_bleu,
                                 token_text=token_item['text'],
                                 code=' '.join(code),
                                 token_code=token_item['code']))
        # res_df=pd.DataFrame(res_dic).T
        # # print(res_df)
        # excel_writer = pd.ExcelWriter(res_path)  # 根据路径savePath打开一个excel写文件
        # res_df.to_excel(excel_writer,header=True,index=True)
        # excel_writer.save()
        with codecs.open(res_path,'w',encoding='utf-8') as f:
            json.dump(res_data,f,indent=4, ensure_ascii=False)
        logging.info("{} have {} paramerters in total".format(self.model_name, sum(x.numel() for x in self.net.parameters() if x.requires_grad)))
        logging.info('>>>>>>>The result has been saved to {}'.format(res_path))

    def _code_ids2tokens(self,code_idss, code_i2w, end_idx):
        return [[code_i2w[idx] for idx in (code_ids[:code_ids.tolist().index(end_idx)]
                                                    if end_idx in code_ids else code_ids)]
                          for code_ids in code_idss]
    
    def _tgt_ids2tokens(self, text_id_np, text_dic, end_idx=0, **kwargs):
        if self.copy:
            text_tokens: list = []
            for j, text_ids in enumerate(text_id_np):
                text_i2w = {**text_dic['text_i2w'], **text_dic['ex_text_i2ws'][j]}
                end_i = text_ids.tolist().index(end_idx) if end_idx in text_ids else len(text_ids)
                text_tokens.append([text_i2w[text_idx] for text_idx in text_ids[:end_i]])
                # if end_i == 0:
                #     print()
        else:
            text_tokens = [[text_dic[idx] for idx in (text_ids[:text_ids.tolist().index(end_idx)]
                                                      if end_idx in text_ids else text_ids)]
                          for text_ids in text_id_np]

        return text_tokens

if __name__ == '__main__':

    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    model = TModel(model_dir=params['model_dir'],
                   model_name=params['model_name'],
                   model_id=params['model_id'],
                   ast_emb_dims=params['ast_emb_dims'],
                   code_emb_dims=params['code_emb_dims'],
                   text_emb_dims=params['text_emb_dims'],
                   ast_hid_dims=params['ast_hid_dims'],
                   ast_gnn_layers=params['ast_gnn_layers'],
                   ast_GNN=params['ast_GNN'],
                   ast_gnn_aggr=params['ast_gnn_aggr'],
                   code_att_layers=params['code_att_layers'],
                   code_att_heads=params['code_att_heads'],
                   code_att_head_dims=params['code_att_head_dims'],
                   code_ff_hid_dims=params['code_ff_hid_dims'],
                   text_att_layers=params['text_att_layers'],
                   text_att_heads=params['text_att_heads'],
                   text_att_head_dims=params['text_att_head_dims'],
                   text_ff_hid_dims=params['text_ff_hid_dims'],
                   drop_rate=params['drop_rate'],
                   copy=params['copy'],
                   pad_idx=params['pad_idx'],
                   train_batch_size=params['train_batch_size'],
                   pred_batch_size=params['pred_batch_size'],
                   max_big_epochs=params['max_big_epochs'],
                   regular_rate=params['regular_rate'],
                   lr_base=params['lr_base'],
                   lr_decay=params['lr_decay'],
                   min_lr_rate=params['min_lr_rate'],
                   warm_big_epochs=params['warm_big_epochs'],
                   early_stop=params['early_stop'],
                   start_valid_epoch=params['start_valid_epoch'],
                   Net=TNet,
                   Dataset=Datasetx,
                   beam_width=params['beam_width'],
                   train_metrics=train_metrics,
                   valid_metric=valid_metric,
                   test_metrics=test_metrics,
                   train_mode=params['train_mode'])

    logging.info('Load data ...')
    # print(train_avail_data_path)
    with codecs.open(train_avail_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with codecs.open(valid_avail_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    with codecs.open(test_avail_data_path, 'rb') as f:
        test_data = pickle.load(f)

    with codecs.open(code_node_i2w_path, 'rb') as f:
        code_i2w = pickle.load(f)

    with codecs.open(test_token_data_path,'r') as f:
        test_token_data=json.load(f)
    # print(len(train_data['texts']), len(valid_data['texts']), len(test_data['texts']))
    model.fit(train_data=train_data,
              valid_data=valid_data)

    for key, value in params.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('Parameters are listed below: \n'+'\n'.join(['{}: {}'.format(key,value) for key,value in params.items()]))

    test_eval_df=model.eval(test_srcs=test_data['code_asts'],
                            test_tgts=test_data['texts'],
                            tgt_i2w=test_data['text_dic'])
    logging.info('Model performance on test dataset:\n')
    for i in range(0,len(test_eval_df.columns),4):
        print(test_eval_df.iloc[:, i:i+4])

    model.generate_texts(code_asts=test_data['code_asts'],
                         text_dic=test_data['text_dic'],
                         res_path=res_path,
                         code_i2w=code_i2w,
                         gold_texts=test_data['texts'],
                         token_data=test_token_data)
