#!/usr/bin/env python3
"""Task 3: 60% prefix + OSM→token embedding (apples-to-apples)."""
import sys, os as _os, math, ast, csv, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}; R=6371000.0
NS=10000; EP=18; BS=256; PA=4; OSM_DIM=5

def pin(l,la): return PORTO['lon_min']<=l<=PORTO['lon_max'] and PORTO['lat_min']<=la<=PORTO['lat_max']
def hav(l1,n1,l2,n2):
    d=math.radians(l2-l1); e=math.radians(n2-n1)
    a=math.sin(d/2)**2+math.cos(math.radians(l1))*math.cos(math.radians(l2))*math.sin(e/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
def hvk(l1,n1,l2,n2): return hav(l1,n1,l2,n2)/1000
def xy(l,la): return l*111320*math.cos(math.radians(41.15)),la*111320

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

class OSM:
    RF={'major':1,'secondary':2,'tertiary':4,'local':8,'nonmotorized':16}
    DO=['major','secondary','tertiary','local','nonmotorized']; DE={n:i+1 for i,n in enumerate(DO)}
    CS=120.0
    def __init__(s,p): s.qc={}; s._l(p)
    @classmethod
    def mhc(s,hw):
        if not hw: return None
        if hw in {'motorway','motorway_link','trunk','trunk_link','primary','primary_link'}: return 'major'
        if hw in {'secondary','secondary_link'}: return 'secondary'
        if hw in {'tertiary','tertiary_link'}: return 'tertiary'
        if hw in {'residential','service','living_street','unclassified'}: return 'local'
        if hw in {'pedestrian','footway','cycleway','path','track','steps'}: return 'nonmotorized'
        return None
    def _ck(s,x,y): return int(x//s.CS),int(y//s.CS)
    def _nb(s,x,y,r):
        hp=max(1,int(math.ceil(r/s.CS))); cx,cy=s._ck(x,y); o=[]
        for dx in range(-hp,hp+1):
            for dy in range(-hp,hp+1): o.extend(s.g.get((cx+dx,cy+dy),[]))
        return np.asarray(o,dtype=np.int32) if o else np.empty(0,dtype=np.int32)
    def _nd(s,x,y,fl,r=1500.0):
        i=s._nb(x,y,r)
        if len(i)==0: return r
        m=(s.m[i]&s.RF[fl])!=0
        if not np.any(m): return r
        dx=s.nx[i[m]]-x; dy=s.ny[i[m]]-y; return float(np.sqrt(np.min(dx*dx+dy*dy)))
    def pf(s,lo,la,pf):
        ck=(pf,round(lo,5),round(la,5))
        if ck in s.qc: return s.qc[ck]
        x,y=xy(lo,la); i2=s._nb(x,y,200.0); i3=s._nb(x,y,300.0)
        def ct(i,rm,fn=None):
            if len(i)==0: return 0
            dx=s.nx[i]-x; dy=s.ny[i]-y; inside=(dx*dx+dy*dy)<=rm*rm
            if fn: inside&=(s.m[i]&s.RF[fn])!=0
            return int(np.sum(inside))
        dc={n:ct(i2,200.0,n) for n in s.DO}
        dt=max(s.DO,key=lambda n:(dc[n],-s.DE[n]))
        de=s.DE[dt] if dc[dt]>0 else 0
        f={f'{pf}_nearest_major_m':s._nd(x,y,'major'),f'{pf}_nearest_secondary_m':s._nd(x,y,'secondary'),
           f'{pf}_local_road_count_r200':ct(i2,200.0,'local'),f'{pf}_major_road_count_r300':ct(i3,300.0,'major'),
           f'{pf}_node_density_r200':ct(i2,200.0)}
        s.qc[ck]=f; return f
    def _l(s,p):
        print(f"  OSM: {p}")
        with open(p,'r',encoding='utf-8') as f: d=json.load(f)
        nc={}; nm={}; wc=0
        for el in d.get('elements',[]):
            if el.get('type')=='node':
                lo=el.get('lon'); la=el.get('lat')
                if lo is None or la is None: continue
                if pin(lo,la): nc[el['id']]=(float(lo),float(la))
        for el in d.get('elements',[]):
            if el.get('type')!='way': continue
            rc=s.mhc((el.get('tags') or {}).get('highway'))
            if rc is None: continue
            fl=s.RF[rc]; v=0
            for nid in el.get('nodes',[]):
                if nid in nc: nm.setdefault(nid,0); nm[nid]|=fl; v+=1
            if v>=2: wc+=1
        kept=[nid for nid,m in nm.items() if m>0 and nid in nc]
        xs,ys,ms=[],[],[]; grid={}
        for i,nid in enumerate(kept):
            lo,la=nc[nid]; x,y=xy(lo,la); xs.append(x); ys.append(y); ms.append(nm[nid])
            grid.setdefault(s._ck(x,y),[]).append(i)
        s.nx=np.asarray(xs,dtype=np.float32); s.ny=np.asarray(ys,dtype=np.float32)
        s.m=np.asarray(ms,dtype=np.int32); s.g=grid
        print(f"  Rdy: {len(kept)} n, {wc} w")

def traj_prefix(pts, ratio):
    n=max(2,int(len(pts)*ratio)); return pts[:n],n

def load(ns):
    print(f"Load {ns}..."); rows=[]
    with open('kaggle_dl/train.csv') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if row['MISSING_DATA']=='True': continue
            try: p=ast.literal_eval(row['POLYLINE'])
            except: continue
            if len(p)<10: continue
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
            rows.append({'poly':p,'ts':int(row['TIMESTAMP']),'call':row.get('CALL_TYPE','C')})
            if len(rows)>=ns: break
    print(f"  {len(rows)}"); return rows

def build_cont(rows, ratio=0.6, osm=None, maxlen=40):
    tk,mt,yo,ofs=[],[],[],[]
    for idx,r in enumerate(rows):
        pref,n=traj_prefix(r['poly'],ratio)
        if len(pref)<5: continue
        # Resample to fixed length
        if len(pref) > maxlen:
            idxs = np.linspace(0, len(pref)-1, maxlen, dtype=int)
            pref = [pref[i] for i in idxs]
        if idx%2000==0: print(f"    {idx}/{len(rows)}",flush=True)
        sl,sa=pref[0]; pl,pa=sl,sa; cm=0; pb=0; ts=[]; osf=[]
        for i,(lo,la) in enumerate(pref):
            dx=(lo-sl)*111320*math.cos(math.radians(41.15)); dy=(la-sa)*111320
            if i==0: sx=sy=sd=sp=0; tb=0
            else:
                sx=(lo-pl)*111320*math.cos(math.radians(41.15)); sy=(la-pa)*111320
                sd=hav(pa,pl,la,lo); sp=sd/15
                cb=math.atan2(la-pa,lo-pl); tb=((cb-pb+math.pi)%(2*math.pi)-math.pi)/math.pi; pb=cb; cm+=sd
            ts.append([dx/1000,dy/1000,sx/1000,sy/1000,sd/1000,sp/10,tb,cm/1000,i/max(1,len(pref)-1)])
            br=math.atan2(dy,dx) if i>0 else 0; ts[-1].extend([math.sin(br),math.cos(br),cm/1000,sd/1000])
            pl,pa=lo,la
            if osm:
                pfx=f'p{i}'; f=osm.pf(lo,la,pfx)
                osf.append([f[f'{pfx}_nearest_major_m']/1500,f[f'{pfx}_nearest_secondary_m']/1500,
                            f[f'{pfx}_local_road_count_r200']/20,f[f'{pfx}_major_road_count_r300']/10,
                            f[f'{pfx}_node_density_r200']/50])
        e=r['poly'][-1]; st=hvk(sa,sl,pref[-1][1],pref[-1][0])*1000
        h=(r['ts']//3600)%24; w=(r['ts']//86400)%7
        tk.append(ts); mt.append([h,w,st/1000,len(pref)/float(max(1,len(pref)*2))]); yo.append([e[1],e[0]])
        if osm: ofs.append(osf)
    # Pad to maxlen
    maxl=max(len(x) for x in tk); tok_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in tk]
    osm_pad=None
    if osm:
        osm_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in ofs]
        osm_pad=np.array(osm_pad,dtype=np.float32)
    return np.array(tok_pad,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(yo,dtype=np.float32),osm_pad

def build_grid(rows, tok, ratio=0.6, osm=None, maxlen=40):
    ids,aux,mt,yo,ofs=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pref,n=traj_prefix(r['poly'],ratio)
        if len(pref)<5: continue
        # Resample to fixed length
        if len(pref) > maxlen:
            idxs = np.linspace(0, len(pref)-1, maxlen, dtype=int)
            pref = [pref[i] for i in idxs]
        if idx%2000==0: print(f"    {idx}/{len(rows)}",flush=True)
        gi,ga,go=[],[],[]
        for i,(lo,la) in enumerate(pref):
            gid,ox,oy=tok.enc(lo,la); gi.append(gid+1); ga.append([ox,oy])
            if osm:
                pfx=f'p{i}'; f=osm.pf(lo,la,pfx)
                go.append([f[f'{pfx}_nearest_major_m']/1500,f[f'{pfx}_nearest_secondary_m']/1500,
                           f[f'{pfx}_local_road_count_r200']/20,f[f'{pfx}_major_road_count_r300']/10,
                           f[f'{pfx}_node_density_r200']/50])
        e=r['poly'][-1]; st=hvk(pref[0][1],pref[0][0],pref[-1][1],pref[-1][0])*1000
        h=(r['ts']//3600)%24; w=(r['ts']//86400)%7
        ids.append(gi); aux.append(ga); mt.append([h,w,st/1000,len(pref)/float(max(1,len(pref)*2))])
        yo.append([e[1],e[0]]); ofs.append(go) if osm else None
    # Pad to maxlen
    maxl=max(len(x) for x in ids)
    ids_pad=[x+[0]*(maxl-len(x)) for x in ids]
    aux_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in aux]
    osm_pad=None
    if osm:
        osm_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in ofs]
        osm_pad=np.array(osm_pad,dtype=np.float32)
    return np.array(ids_pad,dtype=np.int64),np.array(aux_pad,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(yo,dtype=np.float32),osm_pad

def build_clust(rows, tok, ratio=0.6, osm=None, maxlen=40):
    ids,aux,mt,yo,ofs=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pref,n=traj_prefix(r['poly'],ratio)
        if len(pref)<5: continue
        # Resample to fixed length
        if len(pref) > maxlen:
            idxs = np.linspace(0, len(pref)-1, maxlen, dtype=int)
            pref = [pref[i] for i in idxs]
        if idx%2000==0: print(f"    {idx}/{len(rows)}",flush=True)
        ci,ca,co=[],[],[]
        for i,(lo,la) in enumerate(pref):
            cid,rx,ry=tok.enc(lo,la); ci.append(cid+1); ca.append([rx,ry])
            if osm:
                pfx=f'p{i}'; f=osm.pf(lo,la,pfx)
                co.append([f[f'{pfx}_nearest_major_m']/1500,f[f'{pfx}_nearest_secondary_m']/1500,
                           f[f'{pfx}_local_road_count_r200']/20,f[f'{pfx}_major_road_count_r300']/10,
                           f[f'{pfx}_node_density_r200']/50])
        e=r['poly'][-1]; st=hvk(pref[0][1],pref[0][0],pref[-1][1],pref[-1][0])*1000
        h=(r['ts']//3600)%24; w=(r['ts']//86400)%7
        ids.append(ci); aux.append(ca); mt.append([h,w,st/1000,len(pref)/float(max(1,len(pref)*2))])
        yo.append([e[1],e[0]]); ofs.append(co) if osm else None
    # Pad to maxlen
    maxl=max(len(x) for x in ids)
    ids_pad=[x+[0]*(maxl-len(x)) for x in ids]
    aux_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in aux]
    osm_pad=None
    if osm:
        osm_pad=[x+[[0]*len(x[0])]*(maxl-len(x)) for x in ofs]
        osm_pad=np.array(osm_pad,dtype=np.float32)
    return np.array(ids_pad,dtype=np.int64),np.array(aux_pad,dtype=np.float32),np.array(mt,dtype=np.float32),np.array(yo,dtype=np.float32),osm_pad

class CDT(nn.Module):
    def __init__(s,td,md,od,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.tp=nn.Linear(td,d); s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*2,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,2))
        s.uosm=od>0; s.op=nn.Linear(od,d) if od else None
    def forward(s,t,m,o=None):
        x=s.tp(t)
        if s.uosm and o is not None: x=x+s.op(o)
        cls=s.cls.expand(t.size(0),-1,-1); x=torch.cat([cls,x],1); x=x+s.pe[:,:x.size(1),:]
        x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,mf],1))

class TDT(nn.Module):
    def __init__(s,voc,ad,md,od,ml,d=64,nh=4,nl=2):
        super().__init__()
        s.emb=nn.Embedding(voc+1,d,padding_idx=0); s.ap=nn.Linear(ad,d)
        s.mp=nn.Sequential(nn.Linear(md,d),nn.GELU(),nn.LayerNorm(d))
        s.cls=nn.Parameter(torch.zeros(1,1,d)); s.pe=nn.Parameter(torch.zeros(1,ml+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(el,nl); s.norm=nn.LayerNorm(d); s.ps=nn.Linear(d,1)
        s.hd=nn.Sequential(nn.Linear(d*2,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,2))
        s.uosm=od>0; s.op=nn.Linear(od,d) if od else None
    def forward(s,ids,aux,m,o=None):
        x=s.emb(ids)+s.ap(aux)
        if s.uosm and o is not None: x=x+s.op(o)
        cls=s.cls.expand(ids.size(0),-1,-1); x=torch.cat([cls,x],1); x=x+s.pe[:,:x.size(1),:]
        x=s.enc(x); cf=s.norm(x[:,0,:]); tf=s.norm(x[:,1:,:])
        at=torch.softmax(s.ps(tf),1); po=torch.sum(tf*at,1); mf=s.mp(m)
        return s.hd(torch.cat([cf,mf],1))

def herr(p,t):
    l1,n1=np.radians(p[:,0]),np.radians(p[:,1]); l2,n2=np.radians(t[:,0]),np.radians(t[:,1])
    a=np.sin((l2-l1)/2)**2+np.cos(l1)*np.cos(l2)*np.sin((n2-n1)/2)**2
    return R/1000*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tr_c(Xt,Xm,Xo,y,lb,od,ep=EP,bs=BS):
    n=len(Xt); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    ytr,yva,yte=y[tr],y[va],y[te]
    tm=Xt[tr].mean(axis=(0,1)); ts=Xt[tr].std(axis=(0,1))+1e-6; mm=Xm[tr].mean(0); ms=Xm[tr].std(0)+1e-6
    Xt=(Xt-tm)/ts; Xm=(Xm-mm)/ms
    if Xo is not None:
        om=Xo[tr].mean(axis=(0,1)); os_=Xo[tr].std(axis=(0,1))+1e-6; Xo=(Xo-om)/os_
    yt=torch.tensor(ytr).to(DEV); yv=torch.tensor(yva).to(DEV); yte_t=torch.tensor(yte).to(DEV)
    Xtr_t=torch.tensor(Xt[tr]).to(DEV); Xva_t=torch.tensor(Xt[va]).to(DEV); Xte_t=torch.tensor(Xt[te]).to(DEV)
    Xtr_m=torch.tensor(Xm[tr]).to(DEV); Xva_m=torch.tensor(Xm[va]).to(DEV); Xte_m=torch.tensor(Xm[te]).to(DEV)
    Xtr_o=torch.tensor(Xo[tr]).to(DEV) if Xo is not None else None; Xva_o=torch.tensor(Xo[va]).to(DEV) if Xo is not None else None; Xte_o=torch.tensor(Xo[te]).to(DEV) if Xo is not None else None
    m=CDT(Xt.shape[2],Xm.shape[1],od,Xt.shape[1]).to(DEV)
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    cr=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; o_=Xtr_o[b] if Xtr_o is not None else None
            loss=cr(m(Xtr_t[b],Xtr_m[b],o_),yt[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(cr(m(Xva_t,Xva_m,Xva_o),yv).item())
        sc.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if w>=PA: break
    if bs_: m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): pr=m(Xte_t,Xte_m,Xte_o).cpu().numpy()
    er=herr(pr,yte); me=np.mean(er); print(f"  {lb}: {me:.3f} km")
    return me

def tr_t(ids,aux,mt,osm,y,lb,voc,od,ep=EP,bs=BS):
    n=len(ids); idx=np.arange(n); tr,te=train_test_split(idx,test_size=.2,random_state=42); tr,va=train_test_split(tr,test_size=.1,random_state=42)
    ytr,yva,yte=y[tr],y[va],y[te]
    aux=(aux-aux[tr].mean(axis=(0,1)))/(aux[tr].std(axis=(0,1))+1e-6)
    mt=(mt-mt[tr].mean(0))/(mt[tr].std(0)+1e-6)
    if osm is not None:
        om=osm[tr].mean(axis=(0,1)); os_=osm[tr].std(axis=(0,1))+1e-6; osm=(osm-om)/os_
    yt=torch.tensor(ytr).to(DEV); yv=torch.tensor(yva).to(DEV); yte_t=torch.tensor(yte).to(DEV)
    Xtr_i=torch.tensor(ids[tr],dtype=torch.long).to(DEV); Xva_i=torch.tensor(ids[va],dtype=torch.long).to(DEV); Xte_i=torch.tensor(ids[te],dtype=torch.long).to(DEV)
    Xtr_a=torch.tensor(aux[tr]).to(DEV); Xva_a=torch.tensor(aux[va]).to(DEV); Xte_a=torch.tensor(aux[te]).to(DEV)
    Xtr_m=torch.tensor(mt[tr]).to(DEV); Xva_m=torch.tensor(mt[va]).to(DEV); Xte_m=torch.tensor(mt[te]).to(DEV)
    Xtr_o=torch.tensor(osm[tr]).to(DEV) if osm is not None else None; Xva_o=torch.tensor(osm[va]).to(DEV) if osm is not None else None; Xte_o=torch.tensor(osm[te]).to(DEV) if osm is not None else None
    m=TDT(voc,aux.shape[2],mt.shape[1],od,ids.shape[1]).to(DEV)
    o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=2e-4)
    sc=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.6,patience=1)
    cr=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        m.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; o_=Xtr_o[b] if Xtr_o is not None else None
            loss=cr(m(Xtr_i[b],Xtr_a[b],Xtr_m[b],o_),yt[b])
            o.zero_grad();loss.backward();o.step()
        m.eval()
        with torch.no_grad(): vl=float(cr(m(Xva_i,Xva_a,Xva_m,Xva_o),yv).item())
        sc.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if w>=PA: break
    if bs_: m.load_state_dict(bs_); m.eval()
    with torch.no_grad(): pr=m(Xte_i,Xte_a,Xte_m,Xte_o).cpu().numpy()
    er=herr(pr,yte); me=np.mean(er); print(f"  {lb}: {me:.3f} km")
    return me

def main():
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
    rows=load(NS)
    gt=GT(); ct=CT(256)
    ap=[p for r in rows[:int(len(rows)*0.8)] for p in r['poly'][:30]]
    ct.fit(ap)
    print(f"  Grid:{gt.v} Cluster:256")
    print("\nNon-OSM (60% prefix)..."); t0=time.time()
    ct_t,ct_m,ct_y,_=build_cont(rows,ratio=0.6)
    gi,ga,gm,gy,_=build_grid(rows,gt,ratio=0.6)
    ci,ca,cm,cy,_=build_clust(rows,ct,ratio=0.6)
    print(f"  {time.time()-t0:.0f}s  C:{len(ct_t)} G:{len(gi)} Cl:{len(ci)}")

    ose=OSM('data/porto_highways_bbox.osm.json') if _os.path.exists('data/porto_highways_bbox.osm.json') else None
    orow=rows[:min(4000,len(rows))]
    print(f"\nOSM data (60% prefix, {len(orow)} sub)..."); t0=time.time()
    _,_,_,co=build_cont(orow,0.6,ose,40)
    _,_,_,_,go=build_grid(orow,gt,0.6,ose,40)
    _,_,_,_,cso=build_clust(orow,ct,0.6,ose,40)
    print(f"  {time.time()-t0:.0f}s")
    no=min(len(co),len(go),len(cso))

    res={}
    print("\n"+"="*70); print("  TASK 3: 60% PREFIX — FULL 3×2"); print("="*70)
    print("\n── 60% prefix (no OSM) ──")
    res['c']=tr_c(ct_t,ct_m,None,ct_y,"Continuous 60%",0)
    res['g']=tr_t(gi,ga,gm,None,gy,"Grid Token 60%",gt.v,0)
    res['cl']=tr_t(ci,ca,cm,None,cy,"Cluster Token 60%",256,0)
    print("\n── OSM→token emb (60% prefix) ──")
    res['c_sub']=tr_c(ct_t[:no],ct_m[:no],None,ct_y[:no],"Continuous sub",0)
    res['c_osm']=tr_c(ct_t[:no],ct_m[:no],co[:no],ct_y[:no],"Continuous+OSM",OSM_DIM)
    res['g_sub']=tr_t(gi[:no],ga[:no],gm[:no],None,gy[:no],"Grid sub",gt.v,0)
    res['g_osm']=tr_t(gi[:no],ga[:no],gm[:no],go[:no],gy[:no],"Grid+OSM",gt.v,OSM_DIM)
    res['cl_sub']=tr_t(ci[:no],ca[:no],cm[:no],None,cy[:no],"Cluster sub",256,0)
    res['cl_osm']=tr_t(ci[:no],ca[:no],cm[:no],cso[:no],cy[:no],"Cluster+OSM",256,OSM_DIM)

    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    print("  60% prefix:")
    for k,n in [('c','Continuous'),('g','Grid Token'),('cl','Cluster Token')]:
        print(f"    {n:<20s} {res[k]:.3f} km")
    print("  OSM→token emb:")
    for ks,kn,nm in [('c_sub','c_osm','Continuous'),('g_sub','g_osm','Grid'),('cl_sub','cl_osm','Cluster')]:
        nv=res[ks]; ov=res[kn]; d=((ov-nv)/max(nv,1e-6))*100
        print(f"    {nm:<15s} {nv:.3f}→{ov:.3f} ({d:+.2f}%)")

    sp='output/task3_60pct_osm_summary.md'; _os.makedirs('output',exist_ok=True)
    with open(sp,'w') as f:
        f.write("# Task 3: 60% prefix + OSM→token embedding (GPU)\n\n")
        f.write("## Full (no OSM)\n\n")
        for k,n in [('c','Continuous'),('g','Grid Token'),('cl','Cluster Token')]:
            f.write(f"- {n}: {res[k]:.3f} km\n")
        f.write("\n## OSM ablation\n\n")
        f.write("| Method | No OSM | +OSM tok emb | Δ% |\n|---|---:|---:|---:|")
        for ks,kn,nm in [('c_sub','c_osm','Continuous'),('g_sub','g_osm','Grid'),('cl_sub','cl_osm','Cluster')]:
            nv=res[ks]; ov=res[kn]; d=((ov-nv)/max(nv,1e-6))*100
            f.write(f"\n| {nm} | {nv:.3f} | {ov:.3f} | {d:+.2f} |")
    print(f"\nSaved {sp}")

if __name__=="__main__": main()
