import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from benchmark import load_data,rule_predict,evaluate, LABELS

def test_demo_data():
    demo=load_data(ROOT/'data'/'eval_demo.csv')
    assert len(demo)==100
    assert all(sum(r['label']==label for r in demo)==25 for label in LABELS)

def test_keyword_rules():
    assert rule_predict('Tennis player wins championship match')=='Sports'
    assert rule_predict('Parliament debates new border treaty')=='World'
    assert rule_predict('Bank profits rise after quarterly earnings')=='Business'
    assert rule_predict('Scientists discover new telescope signal')=='Sci/Tech'

def test_no_model_leakage():
    train=load_data(ROOT/'data'/'train_demo.csv')
    test=load_data(ROOT/'data'/'eval_demo.csv')
    assert {r['id'] for r in train}.isdisjoint({r['id'] for r in test})
    assert {r['text'] for r in train}.isdisjoint({r['text'] for r in test})

def test_metrics_and_repeats():
    train=load_data(ROOT/'data'/'train_demo.csv')
    rows=load_data(ROOT/'data'/'eval_demo.csv')[:8]
    metrics, attempts=evaluate('nb',train,rows,3,{})
    assert len(attempts)==24
    assert 0<=metrics['accuracy_all_attempts']<=1
    assert metrics['coverage']==1
    assert metrics['strict_repeat_consistency']==1

def test_coverage_when_all_calls_fail():
    train=load_data(ROOT/'data'/'train_demo.csv')
    test=load_data(ROOT/'data'/'eval_demo.csv')[:3]
    # Without an API key, failures must never look like successful classifications.
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {'OPENAI_API_KEY':''}):
        metrics,attempts=evaluate('openai:unavailable-example',train,test,1,{})
    assert metrics['coverage']==0
    assert metrics['accuracy_all_attempts']==0
    assert metrics['api_cost_usd']==''
    assert all(a['error'] for a in attempts)

def test_real_data_sampler_keeps_balance_and_label_mapping():
    from prepare_ag_news import sample
    rows=[]
    for label in range(1,5):
        for i in range(4):
            rows.append([str(label),f'Unique headline {label}/{i}',f'description {i}'])
    selected=sample(rows,2,2026,'test',False)
    assert len(selected)==8
    assert sorted(sum(x['label']==label for x in selected) for label in LABELS)==[2,2,2,2]
