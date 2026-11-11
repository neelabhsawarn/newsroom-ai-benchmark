"""Fetch AG News (source licensing: research/non-commercial) and create disjoint splits.
Run on a machine with internet. Uses upstream official train/test split.
"""
import argparse
import csv
import hashlib
import io
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
LABELS={1:'World',2:'Sports',3:'Business',4:'Sci/Tech'}
BASE='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv'
HASHES={
 'train':'76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e',
 'test':'521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699',
}

def download(split):
    with urllib.request.urlopen(f'{BASE}/{split}.csv',timeout=70) as r:data=r.read()
    checksum=hashlib.sha256(data).hexdigest()
    if checksum!=HASHES[split]:raise ValueError(f'{split} integrity check failed: {checksum}')
    return list(csv.reader(io.StringIO(data.decode('utf-8'))))

def sample(rows,per_label,seed,split,include_description):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if len(r)!=3:raise ValueError(f'Unexpected AG News columns in row {i}: {len(r)}')
        code=int(r[0]); label=LABELS[code]
        title,desc=r[1].strip(),r[2].strip()
        item={'id':f'AG-{split}-{i:06d}','text':title+((' — '+desc) if include_description else ''),'label':label,'source':'AG News '+split+' split'}
        if title:groups[label].append(item)
    rng=random.Random(seed)
    sampled=[]
    for label in LABELS.values():
        if len(groups[label])<per_label:raise ValueError('Not enough samples for '+label)
        sampled.extend(rng.sample(groups[label],per_label))
    rng.shuffle(sampled)
    return sampled

def write(file,rows):
    with file.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','text','label','source']);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-per-class',type=int,default=500)
    p.add_argument('--eval-per-class',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--include-description',action='store_true')
    args=p.parse_args()
    if args.train_per_class<1 or args.eval_per_class<1: p.error('counts must be positive')
    dest=Path(__file__).resolve().parents[1]/'data'
    for split,n in [('train',args.train_per_class),('test',args.eval_per_class)]:
        rows=sample(download(split),n,args.seed,split,args.include_description)
        path=dest/('train_agnews.csv' if split=='train' else 'eval_agnews.csv')
        write(path,rows)
        print(f'{path}: {len(rows)} source-labelled examples')
if __name__=='__main__':main()
