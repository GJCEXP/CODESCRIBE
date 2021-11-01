#coding=utf-8
import json
import pickle
from config import *
from collections import Counter
import sys
import codecs
from tqdm import tqdm
import enchant  # pip install pyenchant
import re
import nltk

def build_w2i2w(train_token_data_path,
                code_node_w2i_path,
                code_node_i2w_path,
                node_pos_w2i_path,
                node_pos_i2w_path,
                text_w2i_path,
                text_i2w_path,
                in_min_token_count=3,
                out_min_token_count=3,
                unk_aliased=True,
                ):
    logging.info('########### Start building the dictionary of the training set ##########')
    dic_paths = [code_node_w2i_path,
                 code_node_i2w_path,
                 node_pos_w2i_path,
                 node_pos_i2w_path,
                 text_w2i_path,
                 text_i2w_path,
                 ]
    for dic_path in dic_paths:
        dic_dir = os.path.dirname(dic_path)
        if not os.path.exists(dic_dir):
            os.makedirs(dic_dir)

    with codecs.open(train_token_data_path, 'r', encoding='utf-8') as f:
        token_data = json.load(f)

    code_node_counter = Counter()
    node_pos_counter = Counter()
    text_token_counter = Counter()
    max_ast_size=0
    max_code_len=0
    max_text_len=0
    for i, item in enumerate(tqdm(token_data)):
        # logging.info('------Process the %d-th item' % (i + 1))
        code_nodes=list(eval(item['ast']['nodes']))+item['code'].split()
        code_node_counter += Counter(code_nodes)
        ast_poses=list(eval(item['ast']['poses']))
        node_pos_counter += Counter(ast_poses)
        text_token_counter += Counter(item['text'].split())  # texts是一个列表
        max_ast_size = max(max_ast_size, len(list(eval(item['ast']['nodes']))))
        max_code_len = max(max_code_len, len(item['code'].split()))
        max_text_len=max(max_text_len,len(item['text'].split()))
    logging.info('max_ast_size: {}, max_code_len: {}, max_text_len: {}'.format(max_ast_size,max_code_len,max_text_len))
    general_vocabs = [PAD_TOKEN, UNK_TOKEN]

    code_nodes = list(filter(lambda x: code_node_counter[x] >= in_min_token_count, code_node_counter.keys()))
    unk_aliases=[]
    if unk_aliased:
        max_alias_num = 0
        for i, item in enumerate(token_data):
            aliases = list(filter(lambda x: x not in code_nodes, set(list(eval(item['ast']['nodes']))+item['code'].split())))
            max_alias_num = max(max_alias_num, len(aliases))
        unk_aliases = ['<unk-alias-{}>'.format(i) for i in range(max_alias_num)]
    code_nodes = general_vocabs + code_nodes+unk_aliases

    node_poses = list(filter(lambda x: node_pos_counter[x] >= in_min_token_count, node_pos_counter.keys()))
    node_poses = general_vocabs + node_poses

    text_tokens = list(filter(lambda x: text_token_counter[x] >= out_min_token_count, text_token_counter.keys()))
    text_tokens = general_vocabs + text_tokens + [OUT_END_TOKEN, OUT_BEGIN_TOKEN,]

    code_node_indices = list(range(len(code_nodes)))
    node_pos_indices = list(range(len(node_poses)))
    text_indices = list(range(len(text_tokens)))

    code_node_w2i = dict(zip(code_nodes, code_node_indices))
    code_node_i2w = dict(zip(code_node_indices, code_nodes))
    node_pos_w2i = dict(zip(node_poses, node_pos_indices))
    node_pos_i2w = dict(zip(node_pos_indices, node_poses))
    text_w2i = dict(zip(text_tokens, text_indices))
    text_i2w = dict(zip(text_indices, text_tokens))

    dics = [code_node_w2i,
            code_node_i2w,
            node_pos_w2i,
            node_pos_i2w,
            text_w2i,
            text_i2w]
    for dic, dic_path in zip(dics, dic_paths):
        with open(dic_path, 'wb') as f:
            pickle.dump(dic, f)
        with codecs.open(dic_path + '.json', 'w') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish building the dictionary of the training set ##########')

def get_ex_tgt_dict(src1_tokens,src2_tokens,tgt_w2i):
    '''
    获取一条target中在两个source中但不在tgt_w2i的token和id之间的映射字典
    :param src1_tokens:
    :param src2_tokens:
    :param tgt_w2i:
    :return:
    '''
    ex_src1_tokens = list(filter(lambda x: x not in tgt_w2i.keys(), src1_tokens))
    ex_src2_tokens = list(filter(lambda x: x not in tgt_w2i.keys(), src2_tokens))
    ex_src_tokens = ex_src1_tokens + ex_src2_tokens
    ex_src_tokens = sorted(list(set(ex_src_tokens)), key=ex_src_tokens.index)  # 去重但保留顺序
    ex_src_token_indices = list(range(len(tgt_w2i), len(tgt_w2i) + len(ex_src_tokens)))
    ex_tgt_w2i = dict(zip(ex_src_tokens, ex_src_token_indices))
    ex_tgt_i2w = dict(zip(ex_src_token_indices, ex_src_tokens))
    return ex_tgt_w2i,ex_tgt_i2w

def get_src2tgt_map_ids(src_tokens,tgt_w2i,ex_tgt_w2i):
    '''
    生成source中的每个token映射为target词库中的id，不再target词库的映射为补充到ex_tgt_w2i的id（序号)
    :param src_tokens:
    :param tgt_w2i:
    :param ex_tgt_w2i:
    :return:
    '''
    # ex_tgt_w2i.update(tgt_w2i)  #不能反过来，否则tgt_w2i会被改变
    all_tgt_w2i = {**tgt_w2i, **ex_tgt_w2i}
    src_map=[all_tgt_w2i[token] for token in src_tokens]
    return src_map

def get_align_tgt_ids(tgt_tokens,tgt_w2i,ex_tgt_w2i):
    '''
    将target中的token映射为补充了ex_tgt_w2i的id
    :param tgt_tokens:
    :param tgt_w2i:
    :param ex_tgt_w2i:
    :return:
    '''
    # ex_tgt_w2i.update(tgt_w2i)  #不能反过来，否则tgt_w2i会被改变
    all_tgt_w2i={**tgt_w2i,**ex_tgt_w2i}
    unk_idx = tgt_w2i[UNK_TOKEN]
    tgt_token_ids=[all_tgt_w2i.get(token,unk_idx) for token in tgt_tokens]
    return tgt_token_ids

def build_avail_data(token_data_path,
                     avail_data_path,
                     code_node_w2i_path,
                     node_pos_w2i_path,
                     text_w2i_path,
                     text_i2w_path,
                     unk_aliased=True):
    '''
    根据字典构建模型可用的数据集，数据集为一个列表，每个元素为一条数据，是由输入和输出两个元素组成的，
    输入元素为一个ndarray，每行分别为边起点、边终点、深度、全局位置、局部位置，
    输出元素为一个ndarray，为输出的后缀表达式
    :param token_data_path:
    :param avail_data_path:
    :param code_node_w2i_path:
    :param edge_depth_w2i_path:
    :param edge_lpos_w2i_path:
    :param edge_spos_w2i_path:
    :return:
    '''
    logging.info('########### Start building the train dataset available for the model ##########')
    avail_data_dir = os.path.dirname(avail_data_path)
    if not os.path.exists(avail_data_dir):
        os.makedirs(avail_data_dir)

    w2is=[]
    for w2i_path in [code_node_w2i_path,
                     node_pos_w2i_path,
                     text_w2i_path,
                     text_i2w_path]:
        with open(w2i_path,'rb') as f:
            w2is.append(pickle.load(f))
    code_node_w2i,node_pos_w2i,text_w2i,text_i2w=w2is

    logging.info('We have {} code and node tokens, {} node_pos tokens, {} text tokens'.
                 format(len(code_node_w2i),len(node_pos_w2i),len(text_w2i)))
    unk_idx = w2is[0][UNK_TOKEN]
    pad_idx=w2is[0][PAD_TOKEN]
    with codecs.open(token_data_path,'r') as f:
        token_data=json.load(f)

    avail_data={'code_asts':[],'texts':[],'ids':[],
                'text_dic':{'text_i2w':text_i2w,'ex_text_i2ws':[]} #每个out有个不同的ex_text_i2w
                }
    text_token_idx_counter=Counter()
    # w2is = [code_node_w2i,
    #         node_pos_w2i,
    #         code_node_w2i,
    #         ]
    max_ast_size = 0
    max_code_len = 0
    max_text_len = 0
    pbar=tqdm(token_data)
    for i,item in enumerate(pbar):
        # logging.info('------Process the %d-th item' % (i+1))
        # token_id_seqs = []
        ast_nodes = list(eval(item['ast']['nodes']))
        ast_poses=list(eval(item['ast']['poses']))
        code_tokens=item['code'].split()
        text_tokens=item['text'].split()

        ex_text_w2i, ex_text_i2w = get_ex_tgt_dict(code_tokens, ast_nodes, text_w2i)
        code2text_map_ids = get_src2tgt_map_ids(code_tokens, text_w2i, ex_text_w2i)
        node2text_map_ids = get_src2tgt_map_ids(ast_nodes, text_w2i, ex_text_w2i)
        text_token_ids = get_align_tgt_ids(text_tokens, text_w2i, ex_text_w2i)
        text_token_idx_counter += Counter(text_token_ids)

        if unk_aliased:
            all_unk_aliases = filter(lambda x: x not in code_node_w2i.keys(), ast_nodes+code_tokens)
            unk_aliases=[]
            for unk_alias in all_unk_aliases:
                if unk_alias not in unk_aliases:
                    unk_aliases.append(unk_alias)
            ast_nodes = [node if node not in unk_aliases else '<unk-alias-{}>'.format(unk_aliases.index(node)) for node in ast_nodes]
            code_tokens= [token if token not in unk_aliases else '<unk-alias-{}>'.format(unk_aliases.index(token)) for token in code_tokens]

        ast_node_ids=[code_node_w2i.get(node,unk_idx) for node in ast_nodes]
        ast_pos_ids=[node_pos_w2i.get(pos,unk_idx) for pos in ast_poses]
        code_token_ids=[code_node_w2i.get(token,unk_idx) for token in code_tokens]

        edges=np.array(eval(item['ast']['edges']))
        reversed_edges=np.array([edges[1,:],edges[0,:]])
        edges=np.concatenate([edges,reversed_edges],axis=-1)    #反相连接也加上

        avail_item_in={'ast': {'nodes': ast_node_ids,
                               'edges': edges,
                               'poses': ast_pos_ids,
                               'src_map': node2text_map_ids},
                       'code': {'tokens': code_token_ids,
                                'src_map': code2text_map_ids}
                       }

        # avail_item_out=avail_item[10]
        avail_data['code_asts'].append(avail_item_in)
        # print(text_token_ids)
        avail_data['texts'].append(text_token_ids)
        avail_data['ids'].append(i)
        avail_data['text_dic']['ex_text_i2ws'].append(ex_text_i2w)

        max_ast_size = max(max_ast_size, len(avail_item_in['ast']['nodes']))
        max_code_len = max(max_code_len, len(avail_item_in['code']['tokens']))
        max_text_len = max(max_text_len, len(text_token_ids))
    logging.info('max_ast_size: {}, max_code_len: {}, max_text_len: {}'.format(max_ast_size,max_code_len,max_text_len))

    logging.info('+++++++++ The ratio of unknown text tokens is:%f' %(text_token_idx_counter[unk_idx]/sum(text_token_idx_counter.values())))
    with open(avail_data_path,'wb') as f:
        pickle.dump(avail_data,f)
    logging.info('########### Finish building the train dataset available for the model ##########')

if __name__=='__main__':
    build_w2i2w(train_token_data_path=train_token_data_path,
                code_node_w2i_path=code_node_w2i_path,
                code_node_i2w_path=code_node_i2w_path,
                node_pos_w2i_path=node_pos_w2i_path,
                node_pos_i2w_path=node_pos_i2w_path,
                text_w2i_path=text_w2i_path,
                text_i2w_path=text_i2w_path,
                in_min_token_count=in_min_token_count,
                out_min_token_count=out_min_token_count,
                unk_aliased=unk_aliased)

    build_avail_data(token_data_path=train_token_data_path,
                     avail_data_path=train_avail_data_path,
                     code_node_w2i_path=code_node_w2i_path,
                     node_pos_w2i_path=node_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     text_i2w_path=text_i2w_path,
                     unk_aliased=unk_aliased)

    build_avail_data(token_data_path=valid_token_data_path,
                     avail_data_path=valid_avail_data_path,
                     code_node_w2i_path=code_node_w2i_path,
                     node_pos_w2i_path=node_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     text_i2w_path=text_i2w_path,
                     unk_aliased=unk_aliased)

    build_avail_data(token_data_path=test_token_data_path,
                     avail_data_path=test_avail_data_path,
                     code_node_w2i_path=code_node_w2i_path,
                     node_pos_w2i_path=node_pos_w2i_path,
                     text_w2i_path=text_w2i_path,
                     text_i2w_path=text_i2w_path,
                     unk_aliased=unk_aliased)
