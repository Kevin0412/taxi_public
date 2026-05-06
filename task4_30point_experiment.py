#!/usr/bin/env python3
"""Task 4: Three tokenization types at UNIFIED 30-point prefix (apples-to-apples)."""
import sys, math, ast, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}; R=6371000.0
SEQ=30; NS=15000; EP=20; BS=256; PA=4

def pin(lon,lat): return PORTO['lon_min']<=lon<=PORTO['lon_max'] and PORTO['lat_min']<=lat<=PORTO['lat_max']
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

def build_c(rows):
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

def build_g(rows,tok):
    ids,au,mt,y=[],[],[],[]
    for r in rows:
        p=r['p'][:SEQ]; gi,ga=[],[]
        for lo,la in p: gid,ox,oy=tok.enc(lo,la); gi.append(gid+1); ga.append([ox,oy])
        ids.append(gi); au.append(ga); mt.append(bm(p,r['ts'])); y.append(r['d'])
    return np.array(ids,dtype=np.int64),np.array(au,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(y,dtype=np.float32)

def build_cl(rows,tok):
    ids,au,mt,y=[],[],[],[]
    for r in rows:
        p=r['p'][:SEQ]; ci,ca=[],[]
        for lo,la in p: cid,rx,ry=tok.enc(lo,la); ci.append(cid+1); ca.append([rx,ry])
        ids.append(ci); ca.append(ca); mt.append(bm(p,r['ts'])); y.append(r['d'])
    return np.array(ids,dtype=np.int64),np.array(au,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(y,dtype=np.float32)

# I found a bug in build_cl - ca.append(ca) should be ca.append(ca). Let me check...
# Actually wait, that's wrong. It should be aux.append(ca) not ca.append(ca)
# Let me just write cleaner code

# ── Re-do with correct variable names ──
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

class CSeqT(nn.Module):
    def __init__(s,tok_d,meta_d,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.tp=nn.Linear(tok_d,d); s.mp=nn.Sequential(nn.Linear(meta_d,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,t,m):
        x=s.tp(t); cls=s.cls.expand(t.size(0),-1,-1); x=torch.cat([cls,x],1)
        x=x+s.pe[:,:x.size(1),:]; x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,po,mf],1)).squeeze(1)

class TSeqT(nn.Module):
    def __init__(s,voc,aux_d,meta_d,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.emb=nn.Embedding(voc+1,d,padding_idx=0); s.ap=nn.Linear(aux_d,d)
        s.mp=nn.Sequential(nn.Linear(meta_d,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(s,ids,aux,m):
        x=s.emb(ids)+s.ap(aux); cls=s.cls.expand(ids.size(0),-1,-1); x=torch.cat([cls,x],1)
        x=x+s.pe[:,:x.size(1),:]; x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,po,mf],1)).squeeze(1)

def tr_c(Xt,Xm,y,lb,ep=EP,bs=BS):
    n=len(Xt); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    yl=np.log1p(y); yl_tr,yl_va,yl_te=yl[tr],yl[va],yl[te]
    tm=Xt[tr].mean(axis=(0,1)); ts=Xt[tr].std(axis=(0,1))+1e-6; mm=Xm[tr].mean(0); ms=Xm[tr].std(0)+1e-6
    Xt=(Xt-tm)/ts; Xm=(Xm-mm)/ms
    Xtr_t=torch.tensor(Xt[tr]); Xva_t=torch.tensor(Xt[va]); Xte_t=torch.tensor(Xt[te])
    Xtr_m=torch.tensor(Xm[tr]); Xva_m=torch.tensor(Xm[va]); Xte_m=torch.tensor(Xm[te])
    yl_tr,yl_va=torch.tensor(yl_tr),torch.tensor(yl_va); yl_te=torch.tensor(yl_te)
    m=CSeqT(Xt.shape[2],Xm.shape[1],Xt.shape[1])
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    cr=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; loss=cr(m(Xtr_t[b],Xtr_m[b]),yl_tr[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(cr(m(Xva_t,Xva_m),yl_va).item())
        sc.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if w>=PA: break
    if bs_: m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): pr=np.expm1(m(Xte_t,Xte_m).cpu().numpy())
    tr=np.expm1(yl_te.cpu().numpy())
    mae=np.mean(np.abs(pr-tr)); rmse=np.sqrt(np.mean((pr-tr)**2)); mape=np.mean(np.abs((pr-tr)/np.maximum(tr,1)))*100
    print(f"  {lb}: MAE={mae:.1f}s RMSE={rmse:.1f}s MAPE={mape:.1f}%")
    return mae,rmse,mape

def tr_t(ids,aux,mt,y,lb,voc,ep=EP,bs=BS):
    n=len(ids); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    yl=np.log1p(y); yl_tr,yl_va,yl_te=yl[tr],yl[va],yl[te]
    aux=(aux-aux[tr].mean(axis=(0,1)))/(aux[tr].std(axis=(0,1))+1e-6)
    mt=(mt-mt[tr].mean(0))/(mt[tr].std(0)+1e-6)
    Xtr_i=torch.tensor(ids[tr],dtype=torch.long); Xva_i=torch.tensor(ids[va],dtype=torch.long); Xte_i=torch.tensor(ids[te],dtype=torch.long)
    Xtr_a=torch.tensor(aux[tr]); Xva_a=torch.tensor(aux[va]); Xte_a=torch.tensor(aux[te])
    Xtr_m=torch.tensor(mt[tr]); Xva_m=torch.tensor(mt[va]); Xte_m=torch.tensor(mt[te])
    yl_tr,yl_va=torch.tensor(yl_tr),torch.tensor(yl_va); yl_te=torch.tensor(yl_te)
    m=TSeqT(voc,aux.shape[2],mt.shape[1],ids.shape[1])
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    cr=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; loss=cr(m(Xtr_i[b],Xtr_a[b],Xtr_m[b]),yl_tr[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(cr(m(Xva_i,Xva_a,Xva_m),yl_va).item())
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
    print("\n"+"="*70)
    print("  UNIFIED 30-point prefix — apples-to-apples")
    print("="*70)
    results={}
    print("\n── 30-point comparison (no OSM) ──")
    results['c']=tr_c(c_t,c_m,c_y,"Continuous 30pt")
    results['g']=tr_t(gi,ga,gm,gy,"Grid 200m 30pt",gt.v)
    results['cl']=tr_t(ci,ca,cm,cy,"Cluster k=256 30pt",256)
    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    print(f"  {'Method':<25s} {'MAE(s)':>8s} {'RMSE(s)':>9s} {'MAPE(%)':>8s}")
    print(f"  {'-'*50}")
    for k,n in [('c','Continuous'),('g','Grid 200m'),('cl','Cluster k=256')]:
        r=results[k]; print(f"  {n:<25s} {r[0]:>8.1f} {r[1]:>9.1f} {r[2]:>8.1f}")
    sp='output/task4_30point_summary.md'; _os.makedirs('output',exist_ok=True)
    with open(sp,'w') as f:
        f.write("# Task 4: Unified 30-point prefix comparison\n\n")
        f.write(f"Config: {SEQ}-point prefix, {NS} samples, {EP} epochs\n\n")
        f.write("| Method | MAE (s) | RMSE (s) | MAPE (%) |\n|---|---:|---:|---:|\n")
        for k,n in [('c','Continuous'),('g','Grid 200m'),('cl','Cluster k=256')]:
            r=results[k]; f.write(f"| {n} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} |\n")
    print(f"  Saved {sp}")

if __name__=="__main__": main()
