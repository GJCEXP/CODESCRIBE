import json
from my_lib.util.eval.translate_metric import get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu
from my_lib.util.eval.translate_metric import get_meteor,get_rouge,get_cider

res_path='../../data/java/result/result.json'
with open(res_path,'r') as f:
    res_data=json.load(f)
gold_texts=[]
pred_texts=[]
# sblues=[]
for item in res_data:
    gold_text=item['gold_text'].split()
    pred_text=item['pred_text'].split()
    gold_texts.append([gold_text])
    pred_texts.append(pred_text)

print(get_sent_bleu.__name__,':',get_sent_bleu(pred_texts,gold_texts))
print(get_sent_bleu.__name__,':',get_meteor(pred_texts,gold_texts))
print(get_sent_bleu.__name__,':',get_rouge(pred_texts,gold_texts))
