#coding=utf-8
import logging
import os
import sys
from my_lib.util.eval.translate_metric import get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_data_name='train_data'
valid_data_name='valid_data'
test_data_name='test_data'

#顶级数据目录
top_data_dir= '../../data/java'

raw_data_dir=os.path.join(top_data_dir,'raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))


max_code_len=317     #285
max_ast_size=535
max_text_len=40

token_data_dir=os.path.join(top_data_dir,'token_data/')
train_token_data_path=os.path.join(token_data_dir,'{}.json'.format(train_data_name))
valid_token_data_path=os.path.join(token_data_dir,'{}.json'.format(valid_data_name))
test_token_data_path=os.path.join(token_data_dir,'{}.json'.format(test_data_name))

USER_WORDS=[('\\','n')]

basic_info_dir=os.path.join(top_data_dir,'basic_info/')
size_info_path=os.path.join(basic_info_dir,'size_info.pkl')
seg_word_dic_path=os.path.join(basic_info_dir,'seg_word_dic.json')
size_info_pdf_path=os.path.join(basic_info_dir,'dist_of_code_ast_and_text_size.pdf')
size_info_png_path=os.path.join(basic_info_dir,'dist_of_code_ast_and_text_size.png')

fine_token_data_dir=os.path.join(top_data_dir,'fine_token_data/')
train_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(train_data_name))
valid_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(valid_data_name))
test_fine_token_data_path=os.path.join(fine_token_data_dir,'{}.json'.format(test_data_name))

w2i2w_dir=os.path.join(top_data_dir,'w2i2w/')
code_node_w2i_path=os.path.join(w2i2w_dir,'code_node_w2i.pkl')
code_node_i2w_path=os.path.join(w2i2w_dir,'code_node_i2w.pkl')
node_pos_w2i_path=os.path.join(w2i2w_dir,'node_pos_w2i.pkl')
node_pos_i2w_path=os.path.join(w2i2w_dir,'node_pos_i2w.pkl')
text_w2i_path=os.path.join(w2i2w_dir,'text_w2i.pkl')
text_i2w_path=os.path.join(w2i2w_dir,'text_i2w.pkl')

in_min_token_count=3
out_min_token_count=3
unk_aliased=True  #是否将未知的rare tokens进行标号处理

avail_data_dir=os.path.join(top_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(train_data_name))
valid_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(valid_data_name))
test_avail_data_path=os.path.join(avail_data_dir,'{}.pkl'.format(test_data_name))

#the path of result in practical prediction
res_dir=os.path.join(top_data_dir,'result/')
res_path=os.path.join(res_dir,'result.json')

OUT_BEGIN_TOKEN='</s>'
OUT_END_TOKEN='</e>'
PAD_TOKEN='<pad>'
UNK_TOKEN='<unk>'

model_dir=os.path.join(top_data_dir,'model/')
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" #"6,7,8,9","0,1,2,3"
import os
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
#

emb_dims = 512
params = dict(model_dir=model_dir,
              model_name='code2text',
              model_id=None,
              ast_emb_dims=emb_dims,
              code_emb_dims=emb_dims,
              text_emb_dims=emb_dims,
              ast_hid_dims=emb_dims,
              ast_gnn_layers=6, #############2080*10
              ast_GNN=SAGEConv,
              ast_gnn_aggr='add',
              code_att_layers=2,
              code_att_heads=8,
              code_att_head_dims=None,
              code_ff_hid_dims=4 * emb_dims,
              text_att_layers=6,
              text_att_heads=8,
              text_att_head_dims=None,
              text_ff_hid_dims=4 * emb_dims,
              drop_rate=0.2,
              copy=True,
              pad_idx=0,
              train_batch_size=96,   #96
              pred_batch_size=3*96,
              max_big_epochs=100,
              early_stop=10,
              regular_rate=1e-5,
              lr_base=5e-4,
              lr_decay=0.95,
              min_lr_rate=0.05,
              warm_big_epochs=3,
              # Net=TNet,
              # Dataset=Datasetx,
              beam_width=5,
              start_valid_epoch=50,
              gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
              train_mode=True)
# from tmp_google_bleu import get_sent_bleu
train_metrics = [get_sent_bleu]
valid_metric = get_sent_bleu
test_metrics = [get_rouge,get_meteor,get_sent_bleu,
                get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4]

import random
import torch
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
seeds=[8,0,7,23,124,1084,87]
seed_torch(seeds[1])