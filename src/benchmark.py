"""Reproducible news-headline topic classification benchmark.

Three offline systems: transparent keyword rules, TF-IDF + MultinomialNB,
TF-IDF + logistic regression. Optional OpenAI API models via --models openai:MODEL.
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
LEXICON = {
 'World': {'government','parliament','minister','president','diplomatic','embassy','treaty','ceasefire','peace','election','referendum','united','border','humanitarian','ambassador','foreign','negotiations','political','civilians','tribunal','refugee','envoy','unrest','opposition'},
 'Sports': {'sports','football','goal','goalkeeper','tennis','cricket','rugby','marathon','athlete','championship','tournament','final','match','league','basketball','swimming','cycling','golfer','golf','hockey','volleyball','boxing','motorsport','race','coach','squad','cup','stadium','penalty','semifinal'},
 'Business': {'bank','shares','profit','earnings','revenue','retail','merger','acquisition','shareholder','investor','investment','financing','inflation','market','trader','company','business','consumer','lender','economic','property','mortgage','prices','sales','costs','franchise','quarterly'},
 'Sci/Tech': {'scientists','researchers','research','laboratory','lab','technology','software','computer','chip','cybersecurity','space','satellite','telescope','astronomers','robot','quantum','genome','battery','ai','sensor','astronomy','cloud','processor','science','algorithm','engineering','engineers','experiment','physics'}
}

def load_data(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    if not records: raise ValueError('Dataset is empty')
    for key in ('id','text','label'):
        if any(key not in row or not row[key].strip() for row in records):
            raise ValueError(f'Dataset missing required nonblank field: {key}')
    if any(r['label'] not in LABELS for r in records):
        raise ValueError('Unrecognised label: allowed '+', '.join(LABELS))
    if len({r['id'] for r in records})!=len(records):
        raise ValueError('Duplicate IDs in dataset')
    return records

def tokenize(text):
    import re
    return re.findall(r"\b[a-z]+\b",text.lower())

def rule_predict(text):
    words=tokenize(text)
    counts={label:sum(w in lex for w in words) for label,lex in LEXICON.items()}
    max_count=max(counts.values())
    if max_count==0: return 'World'  # Documented fallback; likely poor for unseen terms.
    return next(label for label in LABELS if counts[label]==max_count)

def make_model(name,train):
    if name=='keyword': return None
    if name not in {'nb','logreg'}: raise ValueError(f'Unknown offline model: {name}')
    clf = MultinomialNB(alpha=0.5) if name=='nb' else LogisticRegression(max_iter=1000,random_state=2026)
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1),clf)
    model.fit([x['text'] for x in train],[x['label'] for x in train])
    return model

def parse_rates(path):
    if not path: return {}
    with open(path,newline='',encoding='utf-8') as f:
        return {r['model']:(float(r['input_usd_per_million']),float(r['output_usd_per_million']))
                for r in csv.DictReader(f)}

@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI(timeout=35.0,max_retries=1)

def openai_predict(model_id,text):
    # Uses installed OpenAI SDK; no key is ever stored in the repo or output.
    if not os.environ.get('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY is not set')
    client = _openai_client()
    res = client.responses.create(
        model=model_id,
        instructions=('Classify the news headline by its PRIMARY SUBJECT. Reply with ONLY one of: '
                      'World, Sports, Business, Sci/Tech. Do not explain.'),
        input=text,
        max_output_tokens=35,
    )
    tokens_in=getattr(getattr(res,'usage',None),'input_tokens',0) or 0
    tokens_out=getattr(getattr(res,'usage',None),'output_tokens',0) or 0
    return res.output_text.strip(),tokens_in,tokens_out

def predict(name,model,text):
    if name=='keyword': return rule_predict(text),0,0
    if name.startswith('openai:'): return openai_predict(name.split(':',1)[1],text)
    return str(model.predict([text])[0]),0,0

def median_or_zero(a): return float(statistics.median(a)) if a else 0.0

def percentile95(a): return float(np.percentile(a,95)) if a else 0.0

def evaluate(name, train, evaluation, runs, rates):
    model=None if name.startswith('openai:') else make_model(name,train)
    attempts=[]
    for run_idx in range(1,runs+1):
        for row in evaluation:
            start=time.perf_counter()
            tokens_in=tokens_out=0
            err=''
            try:
                pred,tokens_in,tokens_out=predict(name,model,row['text'])
                pred=pred.strip().strip('"\'\x60 ')
                if pred not in LABELS:
                    err=f'Invalid category output: {pred[:100]}'
                    pred=''
            except Exception as exc:
                pred=''
                err=f'{type(exc).__name__}: {str(exc)[:160]}'
            elapsed_ms=(time.perf_counter()-start)*1000
            per_million=rates.get(name.split(':',1)[1],(0.0,0.0)) if name.startswith('openai:') else (0.0,0.0)
            usd=(tokens_in*per_million[0]+tokens_out*per_million[1])/1_000_000
            attempts.append({
                'model':name,'run':run_idx,'id':row['id'],'text':row['text'],
                'actual':row['label'],'predicted':pred,'correct':int(pred==row['label']),
                'latency_ms':round(elapsed_ms,3),'input_tokens':tokens_in,'output_tokens':tokens_out,
                'estimated_api_cost_usd':round(usd,9),'error':err
            })
    truth=[a['actual'] for a in attempts]
    predictions=[a['predicted'] if a['predicted'] else '__FAILURE__' for a in attempts]
    successes=[a for a in attempts if a['predicted'] in LABELS]
    # Failures are counted as wrong in all-item accuracy; failure class is not a positive for any label.
    acc=sum(a['correct'] for a in attempts)/len(attempts)
    # Precision/recall/f1 include all items, treating failures as missed true labels.
    macro_p=precision_score(truth,predictions,labels=LABELS,average='macro',zero_division=0)
    macro_r=recall_score(truth,predictions,labels=LABELS,average='macro',zero_division=0)
    macro_f1=f1_score(truth,predictions,labels=LABELS,average='macro',zero_division=0)
    grouped={r['id']:[] for r in evaluation}
    for a in attempts: grouped[a['id']].append(a['predicted'] or '__FAILURE__')
    consistent=sum(len(set(p))==1 for p in grouped.values())/len(grouped)
    # Reference model cost should be provided for pricing; missing rates show 0 but are NOT free.
    cost_known=(not name.startswith('openai:')) or name.split(':',1)[1] in rates
    metrics={
       'model':name,'items':len(evaluation),'repeats':runs,'attempts':len(attempts),
       'accuracy_all_attempts':round(acc,4), 'macro_precision':round(float(macro_p),4),
       'macro_recall':round(float(macro_r),4),'macro_f1':round(float(macro_f1),4),
       'coverage':round(len(successes)/len(attempts),4),'failure_count':len(attempts)-len(successes),
       'strict_repeat_consistency':round(consistent,4),
       'median_latency_ms':round(median_or_zero([a['latency_ms'] for a in attempts]),3),
       'p95_latency_ms':round(percentile95([a['latency_ms'] for a in attempts]),3),
       'api_cost_usd':round(sum(a['estimated_api_cost_usd'] for a in attempts),7) if cost_known else '',
       'cost_notes': 'No external API fees; compute not priced' if not name.startswith('openai:')
                    else ('Cost estimated using user-provided rates' if cost_known else 'UNKNOWN: supply pricing CSV')
    }
    return metrics,attempts

def write_csv(path,rows,fields=None):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0].keys()),extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def build_report(metrics, attempts, output, eval_path, train_path):
    is_demo='demo' in Path(eval_path).name
    lines=['# Newsroom AI tagging benchmark — run report','',
           f'- Evaluation: {`}{Path(eval_path).name}{`}; training: {`}{Path(train_path).name}{`}',
           f'- Data provenance: **{"original synthetic demo examples" if is_demo else "imported dataset; check provenance and terms before sharing"}**',
           '- Labels: World / Sports / Business / Sci/Tech',
           '- Same test examples used for each approach; ML training set is separate from evaluation set.',
           '- Accuracy and macro recall count missing or invalid outputs as misses. Macro precision/F1 use all attempted examples.',
           '- Repeat consistency: share of test items with exactly the same output on every run.',
           '- Latency is local wall-clock inference latency per item and is hardware/network dependent.',
           '- API cost is estimated only when per-token pricing is configured; offline models have no API fee.',
           '', '| Approach | Accuracy | Macro F1 | Coverage | Consistency | Median latency | p95 latency | API cost |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for m in metrics:
        cost=f"$"+"{m['api_cost_usd']:.5f}" if isinstance(m['api_cost_usd'],(float,int)) else 'unknown'
        lines.append(f"| {m['model']} | {m['accuracy_all_attempts']:.1%} | {m['macro_f1']:.3f} | {m['coverage']:.1%} | {m['strict_repeat_consistency']:.1%} | {m['median_latency_ms']:.2f} ms | {m['p95_latency_ms']:.2f} ms | {cost} |")
    lines.extend(['','## Observed failure examples',''])
    for m in metrics:
        problems=[a for a in attempts if a['model']==m['model'] and a['run']==1 and not a['correct']]
        lines.append(f"### {m['model']}")
        if problems:
            for p in problems[:5]:
                lines.append(f"- {`}{p['id']}{`}: true **{p['actual']}**, predicted **{p['predicted'] or 'NO OUTPUT'}** — {p['text']}."+(f" Error: {p['error']}" if p['error'] else ''))
        else:lines.append('- None observed on first run.')
    lines.extend(['','## Business interpretation','',
      'Accuracy alone is insufficient for publication workflows: measure class-specific errors, invalid output coverage, and human escalation needs. Use the confusion matrix and prediction CSV to inspect mistakes.',
      'For a production pilot, define thresholds per high-risk class, adjudicate ambiguous labels, re-run on recent editorial data, and monitor drift.',
      'A result from a tiny or synthetic sample is **not** a claim of production accuracy, model superiority, or human annotation agreement.',
      '', '## Human review', '',
      'Give {` }data/human_review_blank.csv{` } to a second annotator. They should fill {` }human_label{` } without seeing the original label where practical. Run {` }python src/human_agreement.py{` } after labels are provided. Agreement has not been measured unless a populated report exists.',
      ''])
    (Path(output)/'run_report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    default=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default=str(default/'data'/'train_demo.csv'))
    parser.add_argument('--eval',default=str(default/'data'/'eval_demo.csv'))
    parser.add_argument('--out',default=str(default/'results'/'demo'))
    parser.add_argument('--models',default='keyword,nb,logreg',help='Comma-separated: keyword,nb,logreg,openai:MODEL_ID')
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--rates',help='Optional CSV model,input_usd_per_million,output_usd_per_million')
    args=parser.parse_args()
    if args.runs<1:parser.error('--runs must be >=1')
    train=load_data(args.train); evaluation=load_data(args.eval)
    if {a['id'] for a in train}&{a['id'] for a in evaluation}:
        parser.error('Train and test IDs overlap')
    if {a['text'].strip().lower() for a in train}&{a['text'].strip().lower() for a in evaluation}:
        parser.error('Train and evaluation contain exact duplicate texts')
    models=[x.strip() for x in args.models.split(',') if x.strip()]
    if any(m not in {'keyword','nb','logreg'} and not m.startswith('openai:') for m in models):
        parser.error('Unknown model in --models')
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    rates=parse_rates(args.rates)
    metrics=[]; all_attempts=[]
    for name in models:
        m,attempts=evaluate(name,train,evaluation,args.runs,rates)
        metrics.append(m);all_attempts.extend(attempts)
        print(f"{name}: accuracy={m['accuracy_all_attempts']:.1%}, macroF1={m['macro_f1']:.3f}, coverage={m['coverage']:.1%}, n={m['attempts']}")
    write_csv(output/'metrics.csv',metrics)
    write_csv(output/'predictions.csv',all_attempts)
    first=[a for a in all_attempts if a['run']==1]
    write_csv(output/'error_analysis.csv',[a for a in first if not a['correct']],list(first[0].keys()) if first else None)
    conf=[]
    for name in models:
        subset=[a for a in first if a['model']==name]
        for actual in LABELS:
            for predicted in LABELS+['__FAILURE__']:
                conf.append({'model':name,'actual':actual,'predicted':predicted,
                             'count':sum(a['actual']==actual and (a['predicted'] or '__FAILURE__')==predicted for a in subset)})
    write_csv(output/'confusion_counts.csv',conf)
    build_report(metrics,all_attempts,output,args.eval,args.train)
    print(f'Wrote {output}')
if __name__=='__main__':main()
