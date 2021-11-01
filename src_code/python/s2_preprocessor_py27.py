#coding=utf-8
import json
from config_py27 import *
from my_lib.util.ast_parser import py2ast
from my_lib.util.tokenizer.python_tokenizer import tokenize_python
from my_lib.util.tokenizer.code_tokenizer import tokenize_code
import codecs
import os
from tqdm import tqdm
import re
import numpy as np

def tokenize_raw_data(raw_data_path,token_data_path,seg_word_dic_path,
                      max_code_len=max_code_len,max_text_len=max_text_len,max_ast_size=max_ast_size):
    logging.info('########### Start tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

    token_data_dir = os.path.dirname(token_data_path)
    if not os.path.exists(token_data_dir):
        os.makedirs(token_data_dir)
    with codecs.open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    with codecs.open(seg_word_dic_path,'r', encoding='utf-8') as f:
        seg_word_dic=json.load(f)
    # lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
    real_max_ast_size = 0
    real_max_code_len = 0
    real_max_text_len = 0
    for i,item in enumerate(tqdm(raw_data)):
        # logging.info('------Process the %d-th item' % (i + 1))
        # token_item['id']=item['id']
        text_tokens=tokenize_code(item['text'],user_words=USER_WORDS,lemmatize=True, lower=True,keep_punc=True,
                                  seg_var=True,err_dic=seg_word_dic)
        item['text']=' '.join(text_tokens[:max_text_len])
        item['text']=re.sub(r'\d+','<number>',item['text'])
        nodes,edges,poses=py2ast(item['code'],attr='all',lemmatize=True,lower=True,keep_punc=True,
                                 seg_var=True,err_dic=seg_word_dic)
        # edges=edges[:, :max_ast_size - 1]
        # reversed_edges=np.array([edges[0,:],edges[1,:]])
        # edges=np.concatenate([edges,reversed_edges],axis=-1)    #反相连接也加上
        item['ast']={'nodes':str(nodes[:max_ast_size]),
                     'edges':str(edges[:, :max_ast_size - 1].tolist()),
                     'poses':str(['({},{},{})'.format(pos[0], pos[1], pos[2]) for pos in poses[:max_ast_size]])}
        item['ast']['nodes']=re.sub(r'\d+','<number>',item['ast']['nodes'])
        code_tokens = tokenize_python(item['code'], lower=True,lemmatize=True, keep_punc=True,
                                  seg_var=True,err_dic=seg_word_dic)
        item['code'] = ' '.join(code_tokens[:max_code_len])
        item['code'] = re.sub(r'\d+', '<number>', item['code'])

        real_max_ast_size = max(real_max_ast_size, len(list(eval(item['ast']['nodes']))))
        real_max_code_len = max(real_max_code_len, len(item['code'].split()))
        real_max_text_len = max(real_max_text_len, len(item['text'].split()))
    logging.info('real_max_ast_size: {}, real_max_code_len: {}, real_max_text_len: {}'.
                 format(real_max_ast_size,real_max_code_len,real_max_text_len))

    with codecs.open(token_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

if __name__=='__main__':
    tokenize_raw_data(raw_data_path=train_raw_data_path,
                      token_data_path=train_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_ast_size=max_ast_size)
    tokenize_raw_data(raw_data_path=valid_raw_data_path,
                      token_data_path=valid_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_ast_size=max_ast_size)
    tokenize_raw_data(raw_data_path=test_raw_data_path,
                      token_data_path=test_token_data_path,
                      seg_word_dic_path=seg_word_dic_path,
                      max_text_len=max_text_len,
                      max_code_len=max_code_len,
                      max_ast_size=max_ast_size)