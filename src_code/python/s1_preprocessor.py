#coding=utf-8
import json
from config import *
from my_lib.util.tokenizer.python_tokenizer import tokenize_python
from my_lib.util.tokenizer.code_tokenizer import tokenize_code
import codecs
import os
from tqdm import tqdm
import nltk
import re
import enchant

def _seg_conti_word(word_str, user_words=None):
    """
    Segment a string of word_str using the pyenchant vocabulary.
    Keeps longest possible words that account for all characters,
    and returns list of segmented words.

    :param word_str: (str) The character string to segment.
    :param exclude: (set) A set of string to exclude from consideration.
                    (These have been found previously to lead to dead ends.)
                    If an excluded word occurs later in the string, this
                    function will fail.
    """
    def _seg_with_digit(words):
        text=' '.join(words)
        digits = re.findall(r'\d+', text)
        digits = sorted(list(set(digits)), key=len, reverse=True)
        # digit_str = ''
        for digit in digits:
            text = text.replace(digit, ' ' + digit + ' ')
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        tokens=[lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a') for word
                in text.strip().split()]
        return tokens

    if not user_words:
        user_words = set()
    try:
        eng_dict = enchant.Dict("en_US")
        if eng_dict.check(word_str):
            return [word_str]

        # if not word_str[0].isalpha():  # don't check punctuation etc.; needs more work
        #     return [word_str]

        left_words,right_words = [],[]

        working_chars = word_str
        while working_chars:
            # iterate through segments of the word_str starting with the longest segment possible
            for i in range(len(working_chars), 2, -1):
                left_chars = working_chars[:i]
                if left_chars in user_words or eng_dict.check(left_chars):
                    left_words.append(left_chars)
                    working_chars = working_chars[i:]
                    user_words.add(left_chars)
                    break
            else:
                for i in range(0, len(working_chars) - 2):
                    right_chars = working_chars[i:]
                    if right_chars in user_words or eng_dict.check(right_chars):
                        right_words.insert(0, right_chars)
                        working_chars = working_chars[:i]
                        user_words.add(right_chars)
                        break
                else:
                    return _seg_with_digit(left_words+[working_chars]+right_words)

        if working_chars!='':
            return _seg_with_digit(left_words + [working_chars] + right_words)
        else:
            return _seg_with_digit(left_words + right_words)
    except Exception:
        return _seg_with_digit([word_str])

def make_seg_word_dict(train_raw_data_path,valid_raw_data_path,test_raw_data_path,seg_word_dic_path):
    logging.info('Start making segmented word dictionary.')
    user_words=set(['mk','dir','json','config','html','arange','bool','eval'])
    path_dir = os.path.dirname(seg_word_dic_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(train_raw_data_path,'r') as f1,open(valid_raw_data_path,'r') as f2,open(test_raw_data_path,'r') as f3:
        train_raw_data,valid_raw_data,test_raw_data=json.load(f1),json.load(f2),json.load(f3)
    # token_counter=Counter()
    token_set=set()
    for raw_data in [train_raw_data,valid_raw_data,test_raw_data]:
        pbar = tqdm(raw_data)
        pbar.set_description('[Extract tokens]')
        for item in pbar:
            code_tokens = tokenize_python(item['code'], lower=True, lemmatize=True, keep_punc=True,
                                          seg_var=True,err_dic=None)
            text_tokens = tokenize_code(item['text'], user_words=USER_WORDS, lemmatize=True, lower=True,keep_punc=True,
                                        seg_var=True,err_dic=None)
            token_set |= set(code_tokens)
            token_set |= set(text_tokens)
    seg_token_dic=dict()
    pbar=tqdm(token_set)
    pbar.set_description('[Segment tokens]')
    # seg_count=0
    for token in pbar:
        seg_token=' '.join(_seg_conti_word(token,user_words=user_words))
        if seg_token != token:
            # seg_count+=1
            seg_token_dic[token] = seg_token
            pbar.set_description('[Segment tokens: {}-th segmented]'.format(len(seg_token_dic)))

    with codecs.open(seg_word_dic_path,'w',encoding='utf-8') as f:
        json.dump(seg_token_dic,f,indent=4, ensure_ascii=False)
    logging.info('Finish making segmented word dictionary.')


if __name__=='__main__':
    make_seg_word_dict(train_raw_data_path, valid_raw_data_path, test_raw_data_path, seg_word_dic_path)
