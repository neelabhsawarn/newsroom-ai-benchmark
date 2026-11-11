"""Create blinded annotation sheet and calculate simple human-vs-reference agreement."""
import argparse
import csv
import random
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from benchmark import LABELS, load_data
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eval',default=str(ROOT/'data'/'eval_demo.csv'))
    p.add_argument('--sheet',default=str(ROOT/'data'/'human_review_blank.csv'))
    p.add_argument('--n',type=int,default=24)
    p.add_argument('--score',action='store_true',help='Score filled human_label values')
    a=p.parse_args()
    evaluation=load_data(a.eval)
    if a.score:
        with open(a.sheet,encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        ref={r['id']:r['label'] for r in evaluation}
        done=[r for r in rows if r.get('human_label','').strip()]
        for row in done:
            if row['id'] not in ref:raise ValueError('Unknown item id '+row['id'])
            if row['human_label'].strip() not in LABELS:raise ValueError('Invalid label in '+row['id'])
        if len(done)<2:
            print('Not enough completed items for agreement analysis (at least 2).');return
        y_true=[ref[r['id']] for r in done];y_human=[r['human_label'].strip() for r in done]
        agreement=sum(t==h for t,h in zip(y_true,y_human))/len(done)
        kappa=cohen_kappa_score(y_true,y_human,labels=LABELS)
        print(f'Annotated n={len(done)}; simple agreement={agreement:.1%}; Cohen kappa={kappa:.3f}')
        print('Note: one reviewer versus reference labels, not inter-annotator agreement unless reference labels were independently authored.')
    else:
        selected=random.Random(2026).sample(evaluation,min(a.n,len(evaluation)))
        out=Path(a.sheet);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','text','human_label','notes']);w.writeheader()
            for row in selected:w.writerow({'id':row['id'],'text':row['text'],'human_label':'','notes':''})
        print(f'Created blinded human review sheet at {out}; no human ratings entered')
if __name__=='__main__':main()
