#!/usr/bin/env python3
import argparse, csv, json, re, zipfile
from pathlib import Path
from lxml import html as LH

TIER_NAMES = {0:'Unrated',1:'Bronze V',2:'Bronze IV',3:'Bronze III',4:'Bronze II',5:'Bronze I',6:'Silver V',7:'Silver IV',8:'Silver III',9:'Silver II',10:'Silver I',11:'Gold V',12:'Gold IV',13:'Gold III',14:'Gold II',15:'Gold I',16:'Platinum V',17:'Platinum IV',18:'Platinum III',19:'Platinum II',20:'Platinum I',21:'Diamond V',22:'Diamond IV',23:'Diamond III',24:'Diamond II',25:'Diamond I',26:'Ruby V',27:'Ruby IV',28:'Ruby III',29:'Ruby II',30:'Ruby I'}
RANK_GROUP={0:'unrated', **{i:'bronze' for i in range(1,6)}, **{i:'silver' for i in range(6,11)}, **{i:'gold' for i in range(11,16)}, **{i:'platinum' for i in range(16,21)}, **{i:'diamond' for i in range(21,26)}, **{i:'ruby' for i in range(26,31)}}

def norm_text(s):
    if not s: return ''
    s=s.replace('\xa0',' ')
    s=re.sub(r'[ \t\r\f\v]+',' ',s)
    s=re.sub(r' *\n *','\n',s)
    s=re.sub(r'\n{3,}','\n\n',s)
    return s.strip()

def text_by_id(root, id_):
    els=root.xpath(f'//*[@id="{id_}"]')
    return norm_text(els[0].text_content()) if els else ''

def parse(html_bytes, name):
    doc=html_bytes.decode('utf-8','replace')
    root=LH.fromstring(doc)
    title=''
    ttl=root.xpath('//title/text()')
    t=ttl[0] if ttl else ''
    m=re.search(r'(\d+)번:\s*(.+)', t)
    pid=int(m.group(1)) if m else None
    if m: title=m.group(2).strip()
    te=root.xpath('//*[@id="problem_title"]/text()')
    if te: title=norm_text(te[0])
    if pid is None:
        mm=re.search(r'problem_(\d+)_', name); pid=int(mm.group(1)) if mm else None
    inputs, outputs={},{}
    for pre in root.xpath('//pre[contains(concat(" ", normalize-space(@class), " "), " sampledata ")]'):
        sid=pre.get('id') or ''
        mm=re.search(r'sample-(input|output)-(\d+)', sid)
        if not mm: continue
        kind, idx=mm.group(1), int(mm.group(2))
        txt=pre.text_content()
        if kind=='input': inputs[idx]=txt
        else: outputs[idx]=txt
    samples=[]
    for idx in sorted(set(inputs)|set(outputs)):
        samples.append({'index':idx,'input':inputs.get(idx,''),'output':outputs.get(idx,'')})
    return {'problem_id':pid,'title':title,'url':f'https://www.acmicpc.net/problem/{pid}' if pid else '', 'time_memory_limit':text_by_id(root,'problem_limit'), 'description':text_by_id(root,'problem_description'), 'input_description':text_by_id(root,'problem_input'), 'output_description':text_by_id(root,'problem_output'), 'samples':samples, 'source_file':name}

def load_map(path):
    if not path: return {}
    p=Path(path); mp={}
    if p.suffix in ['.jsonl','.ndjson']:
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                o=json.loads(line); pid=o.get('problemId') or o.get('problem_id') or o.get('id');
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

def apply_diff(p, mp):
    o=mp.get(p['problem_id']); level=None; tags=[]
    if isinstance(o,int): level=o
    elif isinstance(o,str) and o.isdigit(): level=int(o)
    elif isinstance(o,dict):
        for k in ['level','tier','difficulty_index']:
            if k in o and str(o[k]).strip() not in ('','None','null'):
                level=int(o[k]); break
        raw=o.get('tags') or []
        if isinstance(raw,list): tags=[x.get('key') if isinstance(x,dict) else str(x) for x in raw]
    p['difficulty_index']=level
    p['tier_name']=TIER_NAMES.get(level) if level is not None else None
    p['tier_group']=RANK_GROUP.get(level) if level is not None else None
    p['tags']=[x for x in tags if x]
    p['solvedac']=o
    return p

def sft(p):
    sample=''.join([f"\n[예제 입력 {x['index']}]\n{x['input']}\n[예제 출력 {x['index']}]\n{x['output']}" for x in p['samples']])
    user=(f"BOJ {p['problem_id']}번 '{p['title']}' 문제의 solved.ac 난이도를 분류하라.\n\n[제한]\n{p['time_memory_limit']}\n\n[문제]\n{p['description']}\n\n[입력]\n{p['input_description']}\n\n[출력]\n{p['output_description']}{sample}").strip()
    if p['difficulty_index'] is None:
        ans='난이도 정보가 제공되지 않았습니다.'
    else:
        ans=f"difficulty_index: {p['difficulty_index']}\ntier: {p['tier_name']}\ntier_group: {p['tier_group']}"
        if p.get('tags'): ans+='\ntags: '+', '.join(p['tags'])
    return {'problem_id':p['problem_id'],'messages':[{'role':'system','content':'너는 백준 문제를 solved.ac 난이도 체계에 맞게 분류하는 모델이다.'},{'role':'user','content':user},{'role':'assistant','content':ans}]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--difficulty-map'); ap.add_argument('--max-problems',type=int,default=0); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); mp=load_map(args.difficulty_map)
    stats={'total_html':0,'parsed':0,'with_difficulty':0,'without_difficulty':0,'errors':0}
    rows=[]
    raw=(out/'boj_problems_indexed.jsonl').open('w',encoding='utf-8'); sf=(out/'boj_sft_messages.jsonl').open('w',encoding='utf-8'); ids=(out/'problem_ids.txt').open('w',encoding='utf-8')
    with zipfile.ZipFile(args.zip) as zf:
        names=[n for n in zf.namelist() if n.endswith('.html') and '/archive/problem_' in n]
        names.sort(key=lambda n:int(re.search(r'problem_(\d+)_',n).group(1)) if re.search(r'problem_(\d+)_',n) else 10**12)
        for name in names:
            if args.max_problems and stats['parsed']>=args.max_problems: break
            stats['total_html']+=1
            try:
                p=apply_diff(parse(zf.read(name),name),mp)
                if not p['problem_id']: raise ValueError('missing id')
                raw.write(json.dumps(p,ensure_ascii=False)+'\n'); sf.write(json.dumps(sft(p),ensure_ascii=False)+'\n'); ids.write(str(p['problem_id'])+'\n')
                rows.append({'problem_id':p['problem_id'],'title':p['title'],'difficulty_index':p['difficulty_index'],'tier_name':p['tier_name'],'tier_group':p['tier_group'],'description_len':len(p['description']),'sample_count':len(p['samples']),'url':p['url']})
                stats['parsed']+=1; stats['with_difficulty']+= int(p['difficulty_index'] is not None); stats['without_difficulty']+= int(p['difficulty_index'] is None)
            except Exception as e:
                stats['errors']+=1
    raw.close(); sf.close(); ids.close()
    with (out/'boj_problems_indexed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['problem_id','title','difficulty_index','tier_name','tier_group','description_len','sample_count','url']); w.writeheader(); w.writerows(rows)
    (out/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# BOJ LLM preprocessing output\n\nFiles:\n- `boj_problems_indexed.jsonl`: parsed BOJ problem records.\n- `boj_sft_messages.jsonl`: ChatML-style SFT records.\n- `boj_problems_indexed.csv`: compact index.\n- `problem_ids.txt`: problem IDs for solved.ac enrichment.\n- `stats.json`: counts.\n\nDifficulty mapping: solved.ac level 0 = unrated, 1 = Bronze V, ..., 30 = Ruby I. solved.ac states problem levels are contributor-assigned and may change over time, so cache the fetch date when you enrich.\n\nRerun with real solved.ac metadata:\n```bash\npython preprocess_oj_fast.py --zip "OJ(2).zip" --outdir oj_preprocessed --difficulty-map solvedac_levels.jsonl\n```\n''',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
