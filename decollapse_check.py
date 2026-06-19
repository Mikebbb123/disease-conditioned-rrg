import json, re, sys
from collections import Counter

PATS = {
 'cardiomegaly':[r'cardiomegaly',r'enlarged heart',r'cardiac enlargement'],
 'edema':[r'edema',r'vascular congestion',r'cephalization'],
 'consolidation':[r'consolidation',r'consolidative'],
 'atelectasis':[r'atelectasis',r'atelectatic'],
 'effusion':[r'pleural effusion',r'effusion',r'costophrenic blunting'],
}
NEG = r'(no|without|negative for|free of|clear of|or|nor)\b'
def labels(t):
    t=t.lower(); out=set()
    for d,ps in PATS.items():
        for p in ps:
            for m in re.finditer(r'\b'+p+r'\b',t):
                if re.search(NEG+r'[\w ,;]{0,30}$', t[max(0,m.start()-40):m.start()]): continue
                out.add(d); break
    return out

d = json.load(open(sys.argv[1] if len(sys.argv)>1 else 'test_final.json'))
S = d['samples']
preds=[s['prediction'].strip() for s in S]; refs=[s['reference'].strip() for s in S]
pc = Counter(preds); n=len(preds)
print(f'unique preds : {len(pc)}/{n} ({100*len(pc)/n:.1f}%)')
print(f'top-1 share  : {100*pc.most_common(1)[0][1]/n:.1f}%')
ra=sum(len(labels(r))>0 for r in refs); pa=sum(len(labels(p))>0 for p in preds)
print(f'abn rate ref : {100*ra/n:.1f}%   abn rate pred: {100*pa/n:.1f}%')
