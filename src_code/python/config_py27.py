#coding=utf-8
import os
import logging

train_data_name='train_data'
valid_data_name='valid_data'
test_data_name='test_data'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#顶级数据目录
top_data_dir= '../../data/python'

raw_data_dir=os.path.join(top_data_dir,'raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(train_data_name))
valid_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(valid_data_name))
test_raw_data_path=os.path.join(raw_data_dir,'{}.json'.format(test_data_name))

max_code_len=285     #285
max_ast_size=361
max_text_len=22

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

