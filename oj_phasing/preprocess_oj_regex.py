#!/usr/bin/env python3
import argparse,csv,json,re,zipfile,html as HT
from pathlib import Path
TIER_NAMES={0:'Unrated',1:'Bronze V',2:'Bronze IV',3:'Bronze III',4:'Bronze II',5:'Bronze I',6:'Silver V',7:'Silver IV',8:'Silver III',9:'Silver II',10:'Silver I',11:'Gold V',12:'Gold IV',13:'Gold III',14:'Gold II',15:'Gold I',16:'Platinum V',17:'Platinum IV',18:'Platinum III',19:'Platinum II',20:'Platinum I',21:'Diamond V',22:'Diamond IV',23:'Diamond III',24:'Diamond II',25:'Diamond I',26:'Ruby V',27:'Ruby IV',28:'Ruby III',29:'Ruby II',30:'Ruby I'}
RANK_GROUP={0:'unrated', **{i:'bronze' for i in range(1,6)}, **{i:'silver' for i in range(6,11)}, **{i:'gold' for i in range(11,16)}, **{i:'platinum' for i in range(16,21)}, **{i:'diamond' for i in range(21,26)}, **{i:'ruby' for i in range(26,31)}}
TAG_RE=re.compile(r'<[^>]+>')
BR_RE=re.compile(r'<\s*br\s*/?\s*>', re.I)
P_RE=re.compile(r'</\s*(p|div|li|tr|h\d|section|pre)\s*>', re.I)
SCRIPT_RE=re.compile(r'<(script|style)[\s\S]*?</\1>', re.I)

def html_text(x):
    x=SCRIPT_RE.sub('',x)
    x=BR_RE.sub('\n',x); x=P_RE.sub('\n',x)
    x=TAG_RE.sub('',x)
    x=HT.unescape(x).replace('\xa0',' ')
    x=re.sub(r'[ \t\r\f\v]+',' ',x)
    x=re.sub(r' *\n *','\n',x)
    x=re.sub(r'\n{3,}','\n\n',x)
    return x.strip()

def div_by_id(s,id_):
    m=re.search(r'<div\s+id=["\']'+re.escape(id_)+r'["\'][^>]*>([\s\S]*?)</div>\s*</section>',s,re.I)
    if not m:
        m=re.search(r'id=["\']'+re.escape(id_)+r'["\'][^>]*>([\s\S]*?)</div>',s,re.I)
    return html_text(m.group(1)) if m else ''

def parse(b,name):
    s=b.decode('utf-8','replace')
    m=re.search(r'<title>\s*(\d+)번:\s*([^<]+)</title>',s,re.I)
    pid=int(m.group(1)) if m else None; title=HT.unescape(m.group(2)).strip() if m else ''
    mt=re.search(r'id=["\']problem_title["\'][^>]*>([\s\S]*?)</',s,re.I)
    if mt: title=html_text(mt.group(1))
    if pid is None:
        mm=re.search(r'problem_(\d+)_',name); pid=int(mm.group(1)) if mm else None
    inputs={}; outputs={}
    for mm in re.finditer(r'<pre[^>]+id=["\']sample-(input|output)-(\d+)["\'][^>]*>([\s\S]*?)</pre>',s,re.I):
        kind,idx,txt=mm.group(1),int(mm.group(2)),HT.unescape(TAG_RE.sub('',mm.group(3)))
        if kind=='input': inputs[idx]=txt
        else: outputs[idx]=txt
    samples=[{'index':i,'input':inputs.get(i,''),'output':outputs.get(i,'')} for i in sorted(set(inputs)|set(outputs))]
    return {'problem_id':pid,'title':title,'url':f'https://www.acmicpc.net/problem/{pid}' if pid else '', 'time_memory_limit':div_by_id(s,'problem_limit'),'description':div_by_id(s,'problem_description'),'input_description':div_by_id(s,'problem_input'),'output_description':div_by_id(s,'problem_output'),'samples':samples,'source_file':name}

def load_map(path):
    if not path: return {}
    p=Path(path); mp={}
    if p.suffix in ['.jsonl','.ndjson']:
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                o=json.loads(line); pid=o.get('problemId') or o.get('problem_id') or o.get('id')
                if pid: mp[int(pid)]=o
    elif p.suffix=='.json':
        d=json.loads(p.read_text(encoding='utf-8'))
        if isinstance(d,dict):
            for k,v in d.items(): mp[int(k)]=v
        else:
            for o in d:
                pid=o.get('problemId') or o.get('problem_id') or o.get('id')
                if pid: mp[int(pid)]=o
    else:
        with p.open(newline='',encoding='utf-8') as f:
            for o in csv.DictReader(f):
                pid=o.get('problemId') or o.get('problem_id') or o.get('id')
                if pid: mp[int(pid)]=o
    return mp

def diff(p,mp):
    o=mp.get(p['problem_id']); level=None; tags=[]
    if isinstance(o,int): level=o
    elif isinstance(o,str) and o.isdigit(): level=int(o)
    elif isinstance(o,dict):
        for k in ['level','tier','difficulty_index']:
            if k in o and str(o[k]).strip() not in ('','None','null'):
                level=int(o[k]); break
        raw=o.get('tags') or []
        if isinstance(raw,list): tags=[x.get('key') if isinstance(x,dict) else str(x) for x in raw]
    p.update({'difficulty_index':level,'tier_name':TIER_NAMES.get(level) if level is not None else None,'tier_group':RANK_GROUP.get(level) if level is not None else None,'tags':[x for x in tags if x],'solvedac':o})
    return p

def sft(p):
    sample=''.join([f"\n[예제 입력 {x['index']}]\n{x['input']}\n[예제 출력 {x['index']}]\n{x['output']}" for x in p['samples']])
    user=(f"BOJ {p['problem_id']}번 '{p['title']}' 문제의 solved.ac 난이도를 분류하라.\n\n[제한]\n{p['time_memory_limit']}\n\n[문제]\n{p['description']}\n\n[입력]\n{p['input_description']}\n\n[출력]\n{p['output_description']}{sample}").strip()
    ans='난이도 정보가 제공되지 않았습니다.' if p['difficulty_index'] is None else f"difficulty_index: {p['difficulty_index']}\ntier: {p['tier_name']}\ntier_group: {p['tier_group']}"
    return {'problem_id':p['problem_id'],'messages':[{'role':'system','content':'너는 백준 문제를 solved.ac 난이도 체계에 맞게 분류하는 모델이다.'},{'role':'user','content':user},{'role':'assistant','content':ans}]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--difficulty-map'); ap.add_argument('--max-problems',type=int,default=0); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); mp=load_map(args.difficulty_map)
    stats={'total_html':0,'parsed':0,'with_difficulty':0,'without_difficulty':0,'errors':0}; rows=[]
    with zipfile.ZipFile(args.zip) as zf, (out/'boj_problems_indexed.jsonl').open('w',encoding='utf-8') as raw, (out/'boj_sft_messages.jsonl').open('w',encoding='utf-8') as sf, (out/'problem_ids.txt').open('w',encoding='utf-8') as ids:
        names=[n for n in zf.namelist() if n.endswith('.html') and '/archive/problem_' in n]
        names.sort(key=lambda n:int(re.search(r'problem_(\d+)_',n).group(1)) if re.search(r'problem_(\d+)_',n) else 10**12)
        for name in names:
            if args.max_problems and stats['parsed']>=args.max_problems: break
            stats['total_html']+=1
            try:
                p=diff(parse(zf.read(name),name),mp)
                if not p['problem_id']: raise ValueError('missing id')
                raw.write(json.dumps(p,ensure_ascii=False)+'\n'); sf.write(json.dumps(sft(p),ensure_ascii=False)+'\n'); ids.write(str(p['problem_id'])+'\n')
                rows.append({'problem_id':p['problem_id'],'title':p['title'],'difficulty_index':p['difficulty_index'],'tier_name':p['tier_name'],'tier_group':p['tier_group'],'description_len':len(p['description']),'sample_count':len(p['samples']),'url':p['url']})
                stats['parsed']+=1; stats['with_difficulty']+=int(p['difficulty_index'] is not None); stats['without_difficulty']+=int(p['difficulty_index'] is None)
            except Exception: stats['errors']+=1
    with (out/'boj_problems_indexed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['problem_id','title','difficulty_index','tier_name','tier_group','description_len','sample_count','url']); w.writeheader(); w.writerows(rows)
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# BOJ LLM preprocessing output\n\n- `boj_problems_indexed.jsonl`: parsed BOJ records.\n- `boj_sft_messages.jsonl`: ChatML-style SFT records.\n- `boj_problems_indexed.csv`: compact index.\n- `problem_ids.txt`: IDs for solved.ac enrichment.\n- `stats.json`: counts.\n\nDifficulty mapping: solved.ac level 0 = unrated, 1 = Bronze V, ..., 30 = Ruby I. solved.ac help states problem levels are contributor-assigned and may change over time, so cache the fetch date when enriching.\n''',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
