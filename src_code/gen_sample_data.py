import json
raw_data_path='../data/python/raw_data/train_data.json'
with open(raw_data_path,'r') as f:
    raw_data=json.load(f)

raw_data=raw_data[:100]
with open(raw_data_path,'w',encoding='utf-8') as f:
    json.dump(raw_data,f,ensure_ascii=False,indent=4)


