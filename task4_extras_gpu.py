#!/usr/bin/env python3
"""Task 4 extras on GPU: LSTM baseline + weighted loss + verify 30-point results."""
import sys, math, ast, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}; R=6371000.0
SEQ=30; NS=15000; EP=18; BS=256; PA=4; DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pin(l,la): return PORTO['lon_min']<=l<=PORTO['lon_max'] and PORTO['lat_min']<=la<=PORTO['lat_max']
def hav(l1,n1,l2,n2):
    d=math.radians(l2-l1); e=math.radians(n2-n1)
    a=math.sin(d/2)**2+math.cos(math.radians(l1))*math.cos(math.radians(l2))*math.sin(e/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

class GT:
    def __init__(s,c1=0.0018,c2=0.0022):
        s.nl=int(math.ceil(.5/c1)); s.no=int(math.ceil(.4/c2)); s.v=s.nl*s.no; s.c1,s.c2=c1,c2
    def enc(s,lo,la):
        li=int(np.clip(np.floor((la-40.8)/s.c1),0,s.nl-1)); lj=int(np.clip(np.floor((lo+8.8)/s.c2),0,s.no-1))
        return li*s.no+lj,float(np.clip((la-40.8)/s.c1-li,0,.999)),float(np.clip((lo+8.8)/s.c2-lj,0,.999))

class CT:
    def __init__(s,nc=256): s.nc=nc; s.k=None; s.c=None
    def fit(s,p):
        p=np.asarray(p,dtype=np.float32)
        if len(p)>200000: p=p[np.random.choice(len(p),200000,replace=False)]
        s.k=MiniBatchKMeans(n_clusters=s.nc,batch_size=4096,random_state=42,n_init=5).fit(p); s.c=s.k.cluster_centers_.astype(np.float32)
    def enc(s,lo,la):
        cid=int(np.argmin(np.sum((s.c-[lo,la])**2,1))); return cid,float(lo-s.c[cid][0]),float(la-s.c[cid][1])

def load(ns):
    print(f"Load {ns}..."); r=[]
    with open('kaggle_dl/train.csv') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if row['MISSING_DATA']=='True': continue
            try: p=ast.literal_eval(row['POLYLINE'])
            except: continue
            if len(p)<SEQ+2: continue
            if any(not pin(q[0],q[1]) for q in p): continue
            for _ in range(2):
                sp=[hav(p[j-1][1],p[j-1][0],p[j][1],p[j][0])/15*3.6 for j in range(1,len(p))]
                rm=[]
                for j in range(1,len(p)-1):
                    if sp[j-1]>120 and sp[j]>120:
                        ac=hav(p[j-1][1],p[j-1][0],p[j+1][1],p[j+1][0])/15*3.6
                        if ac<84: rm.append(j)
                for j in reversed(rm): p.pop(j)
                if not rm: break
            sp=[hav(p[j-1][1],p[j-1][0],p[j][1],p[j][0])/15*3.6 for j in range(1,len(p))]
            if max(sp)>300 or np.mean([s>120 for s in sp])>.15: continue
            d=(len(p)-1)*15
            if d<60 or d>7200: continue
            r.append({'p':p,'ts':int(row['TIMESTAMP']),'d':float(d)})
            if len(r)>=ns: break
    print(f"  {len(r)}"); return r

def bm(pp,ts):
    h=(ts//3600)%24; w=(ts//86400)%7
    s=[hav(pp[i-1][1],pp[i-1][0],pp[i][1],pp[i][0]) for i in range(1,len(pp))]
    td=sum(s); st=hav(pp[0][1],pp[0][0],pp[-1][1],pp[-1][0])
    ss=[x/15 for x in s]
    return np.array([h,w,int(h in{8,9,17,18,19}),float(len(pp)),td,st,td/max(1,st),
                     np.mean(ss) if ss else 0,np.mean(ss[-3:]) if len(ss)>=3 else (np.mean(ss) if ss else 0)],dtype=np.float32)

def build_cont(rows):
    tk,mt,y=[],[],[]
    for r in rows:
        p=r['p'][:SEQ]; sl,sa=p[0]; pl,pa=sl,sa; cm=0; pb=0; ts=[]
        for i,(lo,la) in enumerate(p):
            dx=(lo-sl)*111320*math.cos(math.radians(41.15)); dy=(la-sa)*111320
            if i==0: sx=sy=sd=sp=0; tb=0
            else:
                sx=(lo-pl)*111320*math.cos(math.radians(41.15)); sy=(la-pa)*111320
                sd=hav(pa,pl,la,lo); sp=sd/15
                cb=math.atan2(la-pa,lo-pl); tb=((cb-pb+math.pi)%(2*math.pi)-math.pi)/math.pi; pb=cb; cm+=sd
            ts.append([dx/1000,dy/1000,sx/1000,sy/1000,sd/1000,sp/10,tb,cm/1000,i/max(1,SEQ-1)])
            br=math.atan2(dy,dx) if i>0 else 0; ts[-1].extend([math.sin(br),math.cos(br),cm/1000,sd/1000])
            pl,pa=lo,la
        tk.append(ts); mt.append(bm(p,r['ts'])); y.append(r['d'])
    return np.array(tk,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(y,dtype=np.float32)

def build_grid(rows,tok):
    ids,aux,mt,y=[],[],[],[]
    for r in rows:
        p=r['p'][:SEQ]; gi,ga=[],[]
        for lo,la in p: gid,ox,oy=tok.enc(lo,la); gi.append(gid+1); ga.append([ox,oy])
        ids.append(gi); aux.append(ga); mt.append(bm(p,r['ts'])); y.append(r['d'])
    return np.array(ids,dtype=np.int64),np.array(aux,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(y,dtype=np.float32)

def build_clust(rows,tok):
    ids,aux,mt,y=[],[],[],[]
    for r in rows:
        p=r['p'][:SEQ]; ci,ca=[],[]
        for lo,la in p: cid,rx,ry=tok.enc(lo,la); ci.append(cid+1); ca.append([rx,ry])
        ids.append(ci); aux.append(ca); mt.append(bm(p,r['ts'])); y.append(r['d'])
    return np.array(ids,dtype=np.int64),np.array(aux,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(y,dtype=np.float32)

# ── Models ───────────────────────────────────────────────────────

# 1. Standard Transformer (verify 30pt results)
class TSeqT(nn.Module):
    def __init__(s,voc,ad,md,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.emb=nn.Embedding(voc+1,d,padding_idx=0); s.ap=nn.Linear(ad,d)
        s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,ids,aux,m):
        x=s.emb(ids)+s.ap(aux); cls=s.cls.expand(ids.size(0),-1,-1); x=torch.cat([cls,x],1)
        x=x+s.pe[:,:x.size(1),:]; x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,po,mf],1)).squeeze(1)

class CSeqT(nn.Module):
    def __init__(s,td,md,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.tp=nn.Linear(td,d); s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,t,m):
        x=s.tp(t); cls=s.cls.expand(t.size(0),-1,-1); x=torch.cat([cls,x],1)
        x=x+s.pe[:,:x.size(1),:]; x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,po,mf],1)).squeeze(1)

# 2. LSTM baseline (replaces Transformer encoder with LSTM)
class CLSTM(nn.Module):
    def __init__(s,td,md,ml,d=64,nl=2):
        super().__init__()
        s.tp=nn.Linear(td,d); s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.lstm=nn.LSTM(d,d,nl,batch_first=True,dropout=0.12,bidirectional=True)
        s.norm=nn.LayerNorm(d*2); s.ps=nn.Linear(d*2,1)
        s.hd=nn.Sequential(nn.Linear(d*2+d,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,t,m):
        x=s.tp(t); o,_=s.lstm(x); o=s.norm(o)
        at=torch.softmax(s.ps(o),1); po=torch.sum(o*at,1); mf=s.mp(m)
        return s.hd(torch.cat([po,mf],1)).squeeze(1)

class TLSTM(nn.Module):
    def __init__(s,voc,ad,md,ml,d=64,nl=2):
        super().__init__()
        s.emb=nn.Embedding(voc+1,d,padding_idx=0); s.ap=nn.Linear(ad,d)
        s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.lstm=nn.LSTM(d,d,nl,batch_first=True,dropout=0.12,bidirectional=True)
        s.norm=nn.LayerNorm(d*2); s.ps=nn.Linear(d*2,1)
        s.hd=nn.Sequential(nn.Linear(d*2+d,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,ids,aux,m):
        x=s.emb(ids)+s.ap(aux); o,_=s.lstm(x); o=s.norm(o)
        at=torch.softmax(s.ps(o),1); po=torch.sum(o*at,1); mf=s.mp(m)
        return s.hd(torch.cat([po,mf],1)).squeeze(1)

# ── Training ─────────────────────────────────────────────────────
def weighted_smooth_l1(pred, target, beta=0.2):
    """Weighted SmoothL1: more weight on long trips."""
    diff = torch.abs(pred - target)
    w = 1.0 + target / 1000.0  # weight proportional to duration
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return (loss * w).mean()

def train(Xt,Xm,y,lb,model_fn,model_args,crit=nn.SmoothL1Loss(beta=0.2),ep=EP,bs=BS):
    n=len(Xt); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    yl=np.log1p(y); yl_tr,yl_va,yl_te=yl[tr],yl[va],yl[te]
    Xt=(Xt-Xt[tr].mean(axis=(0,1)))/(Xt[tr].std(axis=(0,1))+1e-6)
    Xm=(Xm-Xm[tr].mean(0))/(Xm[tr].std(0)+1e-6)
    Xtr_t=torch.tensor(Xt[tr]).to(DEV); Xva_t=torch.tensor(Xt[va]).to(DEV); Xte_t=torch.tensor(Xt[te]).to(DEV)
    Xtr_m=torch.tensor(Xm[tr]).to(DEV); Xva_m=torch.tensor(Xm[va]).to(DEV); Xte_m=torch.tensor(Xm[te]).to(DEV)
    yl_tr=torch.tensor(yl_tr).to(DEV); yl_va=torch.tensor(yl_va).to(DEV); yl_te=torch.tensor(yl_te).to(DEV)
    Xtr=[Xtr_t,Xtr_m]; Xva=[Xva_t,Xva_m]; Xte=[Xte_t,Xte_m]
    m=model_fn(*model_args).to(DEV)
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; pr=m(*[x[b] for x in Xtr])
            loss=crit(pr,yl_tr[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(crit(m(*Xva),yl_va).item())
        sc.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if w>=PA: break
    if bs_: m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): pr=np.expm1(m(*Xte).cpu().numpy())
    tr=np.expm1(yl_te.cpu().numpy())
    mae=np.mean(np.abs(pr-tr)); rmse=np.sqrt(np.mean((pr-tr)**2)); mape=np.mean(np.abs((pr-tr)/np.maximum(tr,1)))*100
    print(f"  {lb}: MAE={mae:.1f}s RMSE={rmse:.1f}s MAPE={mape:.1f}%")
    return mae,rmse,mape

def train_tok(ids,aux,mt,y,lb,model_fn,model_args,crit=nn.SmoothL1Loss(beta=0.2),ep=EP,bs=BS):
    n=len(ids); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    yl=np.log1p(y); yl_tr,yl_va,yl_te=yl[tr],yl[va],yl[te]
    aux=(aux-aux[tr].mean(axis=(0,1)))/(aux[tr].std(axis=(0,1))+1e-6)
    mt=(mt-mt[tr].mean(0))/(mt[tr].std(0)+1e-6)
    Xtr_i=torch.tensor(ids[tr],dtype=torch.long).to(DEV); Xva_i=torch.tensor(ids[va],dtype=torch.long).to(DEV); Xte_i=torch.tensor(ids[te],dtype=torch.long).to(DEV)
    Xtr_a=torch.tensor(aux[tr]).to(DEV); Xva_a=torch.tensor(aux[va]).to(DEV); Xte_a=torch.tensor(aux[te]).to(DEV)
    Xtr_m=torch.tensor(mt[tr]).to(DEV); Xva_m=torch.tensor(mt[va]).to(DEV); Xte_m=torch.tensor(mt[te]).to(DEV)
    yl_tr=torch.tensor(yl_tr).to(DEV); yl_va=torch.tensor(yl_va).to(DEV); yl_te=torch.tensor(yl_te).to(DEV)
    m=model_fn(*model_args).to(DEV)
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; pr=m(Xtr_i[b],Xtr_a[b],Xtr_m[b])
            loss=crit(pr,yl_tr[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(crit(m(Xva_i,Xva_a,Xva_m),yl_va).item())
        sc.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if w>=PA: break
    if bs_: m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): pr=np.expm1(m(Xte_i,Xte_a,Xte_m).cpu().numpy())
    tr=np.expm1(yl_te.cpu().numpy())
    mae=np.mean(np.abs(pr-tr)); rmse=np.sqrt(np.mean((pr-tr)**2)); mape=np.mean(np.abs((pr-tr)/np.maximum(tr,1)))*100
    print(f"  {lb}: MAE={mae:.1f}s RMSE={rmse:.1f}s MAPE={mape:.1f}%")
    return mae,rmse,mape

def main():
    import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
    print(f"Device: {DEV}")
    rows=load(NS)
    gt=GT(); ct=CT(256)
    tp=[p for r in rows[:int(len(rows)*0.8)] for p in r['p'][:SEQ]]
    ct.fit(tp)
    print(f"  Grid:{gt.v}, Cluster:256")
    print("\nBuilding 30-point datasets...")
    t0=time.time()
    c_t,c_m,c_y=build_cont(rows)
    gi,ga,gm,gy=build_grid(rows,gt)
    ci,ca,cm,cy=build_clust(rows,ct)
    print(f"  {time.time()-t0:.0f}s  C:{len(c_t)} G:{len(gi)} Cl:{len(ci)}")

    res={}
    # 1. Verify Transformer 30pt (baseline)
    print("\n"+"="*70); print("  1. TRANSFORMER 30pt (verify)"); print("="*70)
    res['tr_c']=train(c_t,c_m,c_y,"Continuous 30pt",CSeqT,[13,c_m.shape[1],SEQ])
    res['tr_g']=train_tok(gi,ga,gm,gy,"Grid 200m 30pt",TSeqT,[gt.v,2,gm.shape[1],SEQ])
    res['tr_cl']=train_tok(ci,ca,cm,cy,"Cluster 256 30pt",TSeqT,[256,2,cm.shape[1],SEQ])

    # 2. LSTM baseline
    print("\n"+"="*70); print("  2. LSTM BASELINE (30pt)"); print("="*70)
    res['lstm_c']=train(c_t,c_m,c_y,"LSTM Continuous",CLSTM,[13,c_m.shape[1],SEQ])
    res['lstm_g']=train_tok(gi,ga,gm,gy,"LSTM Grid 200m",TLSTM,[gt.v,2,gm.shape[1],SEQ])
    res['lstm_cl']=train_tok(ci,ca,cm,cy,"LSTM Cluster 256",TLSTM,[256,2,cm.shape[1],SEQ])

    # 3. Weighted loss on best config (Cluster 256)
    print("\n"+"="*70); print("  3. WEIGHTED LOSS — Cluster k=256 30pt"); print("="*70)
    res['wl_cl']=train_tok(ci,ca,cm,cy,"Cluster + WeightedLoss",TSeqT,[256,2,cm.shape[1],SEQ],crit=weighted_smooth_l1)

    # Summary
    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    print(f"  {'Method':<30s} {'MAE(s)':>8s} {'RMSE(s)':>9s} {'MAPE(%)':>8s}")
    print(f"  {'-'*55}")
    for k,n in [('tr_c','Transformer Continuous'),('tr_g','Transformer Grid'),('tr_cl','Transformer Cluster 256'),
                ('lstm_c','LSTM Continuous'),('lstm_g','LSTM Grid'),('lstm_cl','LSTM Cluster 256'),
                ('wl_cl','Cluster + WeightedLoss')]:
        r=res[k]; print(f"  {n:<30s} {r[0]:>8.1f} {r[1]:>9.1f} {r[2]:>8.1f}")

    sp='output/task4_extras_summary.md'; _os.makedirs('output',exist_ok=True)
    with open(sp,'w') as f:
        f.write("# Task 4: Extras — LSTM + Weighted Loss\n\n")
        f.write(f"Device: {DEV}, 30pt prefix, {NS} samples\n\n")
        f.write("| Method | MAE (s) | RMSE (s) | MAPE (%) |\n|---|---:|---:|---:|\n")
        for k,n in [('tr_c','Transformer Continuous'),('tr_g','Transformer Grid'),('tr_cl','Transformer Cluster 256'),
                    ('lstm_c','LSTM Continuous'),('lstm_g','LSTM Grid'),('lstm_cl','LSTM Cluster 256'),
                    ('wl_cl','Cluster + WeightedLoss')]:
            r=res[k]; f.write(f"| {n} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} |\n")
    print(f"\nSaved {sp}")

if __name__=="__main__": main()
