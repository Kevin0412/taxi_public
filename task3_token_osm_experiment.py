#!/usr/bin/env python3
"""Task 3: Destination prediction — 3 tokenization × 2 OSM (token embedding) cross experiment."""
import sys, math, ast, csv, json, time

# import os module under an alias to avoid name clashes with local variable 'os' in main()
import os as _os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}
R = 6371000.0
SEQ_POINTS = 20
N_SAMPLE = 10000
N_OSM_SAMPLE = 8000
EPOCHS = 18; BS = 256; PATIENCE = 4
OSM_KEYS = ['nearest_major_m','nearest_secondary_m','local_road_count_r200','major_road_count_r300','node_density_r200']

def point_in_porto(lon,lat): return PORTO['lon_min']<=lon<=PORTO['lon_max'] and PORTO['lat_min']<=lat<=PORTO['lat_max']
def haversine(lat1,lon1,lat2,lon2):
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))
def haversine_km(lat1,lon1,lat2,lon2): return haversine(lat1,lon1,lat2,lon2)/1000.0
def lonlat_xy(lon,lat): return lon*111320*math.cos(math.radians(41.15)), lat*111320

class GridTokenizer:
    def __init__(self, cl=0.0018, co=0.0022):
        self.nl=int(math.ceil((41.3-40.8)/cl)); self.nlo=int(math.ceil((-8.4+8.8)/co))
        self.vocab=self.nl*self.nlo; self.cl,self.co=cl,co
    def encode(self,lon,lat):
        li=int(np.clip(np.floor((lat-40.8)/self.cl),0,self.nl-1))
        lj=int(np.clip(np.floor((lon+8.8)/self.co),0,self.nlo-1))
        return li*self.nlo+lj,float(np.clip((lat-40.8)/self.cl-li,0,0.999)),float(np.clip((lon+8.8)/self.co-lj,0,0.999))

class ClusterTokenizer:
    def __init__(self, nc=256): self.nc=nc; self.kmeans=None; self.centers=None
    def fit(self,pts):
        pts=np.asarray(pts,dtype=np.float32)
        if len(pts)>200000: pts=pts[np.random.choice(len(pts),200000,replace=False)]
        self.kmeans=MiniBatchKMeans(n_clusters=self.nc,batch_size=4096,random_state=42,n_init=5).fit(pts)
        self.centers=self.kmeans.cluster_centers_.astype(np.float32)
    def encode(self,lon,lat):
        cid=int(np.argmin(np.sum((self.centers-[lon,lat])**2,axis=1)))
        return cid,float(lon-self.centers[cid][0]),float(lat-self.centers[cid][1])

class OSMFeatureExtractor:
    ROAD_FLAGS = {'major':1,'secondary':2,'tertiary':4,'local':8,'nonmotorized':16}
    DOMINANT_ORDER = ['major','secondary','tertiary','local','nonmotorized']
    DOMINANT_ENC = {n:i+1 for i,n in enumerate(DOMINANT_ORDER)}
    CELL_SIZE_M = 120.0
    def __init__(self, path): self.query_cache={}; self._load(path)
    @classmethod
    def map_highway_class(cls, hw):
        if not hw: return None
        if hw in {'motorway','motorway_link','trunk','trunk_link','primary','primary_link'}: return 'major'
        if hw in {'secondary','secondary_link'}: return 'secondary'
        if hw in {'tertiary','tertiary_link'}: return 'tertiary'
        if hw in {'residential','service','living_street','unclassified'}: return 'local'
        if hw in {'pedestrian','footway','cycleway','path','track','steps'}: return 'nonmotorized'
        return None
    def _cell_key(self,x,y): return int(x//self.CELL_SIZE_M),int(y//self.CELL_SIZE_M)
    def _nb_idx(self,x,y,r):
        hop=max(1,int(math.ceil(r/self.CELL_SIZE_M))); cx,cy=self._cell_key(x,y); out=[]
        for dx in range(-hop,hop+1):
            for dy in range(-hop,hop+1): out.extend(self.grid.get((cx+dx,cy+dy),[]))
        return np.asarray(out,dtype=np.int32) if out else np.empty(0,dtype=np.int32)
    def _ndist(self,x,y,flag,r=1500.0):
        idx=self._nb_idx(x,y,r)
        if len(idx)==0: return r
        mask=(self.m[idx]&self.ROAD_FLAGS[flag])!=0
        if not np.any(mask): return r
        dx=self.nx[idx[mask]]-x; dy=self.ny[idx[mask]]-y
        return float(np.sqrt(np.min(dx*dx+dy*dy)))
    def point_features(self,lon,lat,pfx):
        ck=(pfx,round(lon,5),round(lat,5))
        if ck in self.query_cache: return self.query_cache[ck]
        x,y=lonlat_xy(lon,lat); i200=self._nb_idx(x,y,200.0); i300=self._nb_idx(x,y,300.0)
        def cnt(idx,rm,fn=None):
            if len(idx)==0: return 0
            dx=self.nx[idx]-x; dy=self.ny[idx]-y; inside=(dx*dx+dy*dy)<=rm*rm
            if fn: inside&=(self.m[idx]&self.ROAD_FLAGS[fn])!=0
            return int(np.sum(inside))
        dc={n:cnt(i200,200.0,n) for n in self.DOMINANT_ORDER}
        dt=max(self.DOMINANT_ORDER,key=lambda n:(dc[n],-self.DOMINANT_ENC[n]))
        de=self.DOMINANT_ENC[dt] if dc[dt]>0 else 0
        feat={f'{pfx}_nearest_major_m':self._ndist(x,y,'major'),
              f'{pfx}_nearest_secondary_m':self._ndist(x,y,'secondary'),
              f'{pfx}_local_road_count_r200':cnt(i200,200.0,'local'),
              f'{pfx}_major_road_count_r300':cnt(i300,300.0,'major'),
              f'{pfx}_node_density_r200':cnt(i200,200.0)}
        self.query_cache[ck]=feat; return feat
    def _load(self,path):
        print(f"  OSM: {path}"); self.path=path
        with open(path,'r',encoding='utf-8') as f: data=json.load(f)
        nc={}; nm={}; wc=0
        for el in data.get('elements',[]):
            if el.get('type')=='node':
                lo=el.get('lon'); la=el.get('lat')
                if lo is None or la is None: continue
                if point_in_porto(lo,la): nc[el['id']]=(float(lo),float(la))
        for el in data.get('elements',[]):
            if el.get('type')!='way': continue
            rc=self.map_highway_class((el.get('tags') or {}).get('highway'))
            if rc is None: continue
            fl=self.ROAD_FLAGS[rc]; v=0
            for nid in el.get('nodes',[]):
                if nid in nc: nm.setdefault(nid,0); nm[nid]|=fl; v+=1
            if v>=2: wc+=1
        kept=[nid for nid,m in nm.items() if m>0 and nid in nc]
        xs,ys,ms=[],[],[]; grid={}
        for i,nid in enumerate(kept):
            lo,la=nc[nid]; x,y=lonlat_xy(lo,la); xs.append(x); ys.append(y); ms.append(nm[nid])
            grid.setdefault(self._cell_key(x,y),[]).append(i)
        self.nx=np.asarray(xs,dtype=np.float32); self.ny=np.asarray(ys,dtype=np.float32)
        self.m=np.asarray(ms,dtype=np.int32); self.grid=grid
        self.nc=len(kept); self.wc=wc
        print(f"  Ready: {self.nc} nodes, {self.wc} ways")

def load_clean(ns):
    print(f"Load {ns}..."); rows=[]
    with open('kaggle_dl/train.csv') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if row['MISSING_DATA']=='True': continue
            try: pts=ast.literal_eval(row['POLYLINE'])
            except: continue
            if len(pts)<10: continue
            if any(not point_in_porto(p[0],p[1]) for p in pts): continue
            for _ in range(2):
                sp=[haversine(pts[j-1][1],pts[j-1][0],pts[j][1],pts[j][0])/15*3.6 for j in range(1,len(pts))]
                rm=[]
                for j in range(1,len(pts)-1):
                    if sp[j-1]>120 and sp[j]>120:
                        ac=haversine(pts[j-1][1],pts[j-1][0],pts[j+1][1],pts[j+1][0])/15*3.6
                        if ac<84: rm.append(j)
                for j in reversed(rm): pts.pop(j)
                if not rm: break
            sp=[haversine(pts[j-1][1],pts[j-1][0],pts[j][1],pts[j][0])/15*3.6 for j in range(1,len(pts))]
            if max(sp)>300: continue
            if np.mean([s>120 for s in sp])>0.15: continue
            if (len(pts)-1)*15<60 or (len(pts)-1)*15>7200: continue
            rows.append({'poly':pts,'ts':int(row['TIMESTAMP']),'call':row.get('CALL_TYPE','C')})
            if len(rows)>=ns: break
    print(f"  Cleaned: {len(rows)}"); return rows

CONT_DIM = 13

def build_cont_ds(rows, osm_ext=None, label=''):
    tok,osm_feats,meta,y=[],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:SEQ_POINTS]
        if len(pts)<SEQ_POINTS: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        slon,slat=pts[0]; plon,plat=slon,slat; cum=0; pb=0; tos=[]
        ofs=[]
        for i,(lon,lat) in enumerate(pts):
            dx=(lon-slon)*111320*math.cos(math.radians(41.15)); dy=(lat-slat)*111320
            if i==0: sx=sy=sd=sp=0; tb=0
            else:
                sx=(lon-plon)*111320*math.cos(math.radians(41.15)); sy=(lat-plat)*111320
                sd=haversine(plat,plon,lat,lon); sp=sd/15
                cb=math.atan2(lat-plat,lon-plon)
                tb=((cb-pb+math.pi)%(2*math.pi)-math.pi)/math.pi; pb=cb; cum+=sd
            tos.append([dx/1000,dy/1000,sx/1000,sy/1000,sd/1000,sp/10,tb,cum/1000,i/max(1,SEQ_POINTS-1)])
            br=math.atan2(dy,dx) if i>0 else 0
            tos[-1].extend([math.sin(br),math.cos(br),cum/1000,sd/1000])
            plon,plat=lon,lat
            if osm_ext:
                pfx=f'p{i}'; fe=osm_ext.point_features(lon,lat,pfx)
                ofs.append([fe[f'{pfx}_nearest_major_m']/1500.0,fe[f'{pfx}_nearest_secondary_m']/1500.0,
                            fe[f'{pfx}_local_road_count_r200']/20.0,fe[f'{pfx}_major_road_count_r300']/10.0,
                            fe[f'{pfx}_node_density_r200']/50.0])
        true_end=r['poly'][-1]
        strt=haversine_km(slat,slon,pts[-1][1],pts[-1][0])*1000
        tok.append(tos); meta.append([(r['ts']//3600)%24,(r['ts']//86400)%7,strt/1000,len(pts)/float(SEQ_POINTS)])
        y.append([true_end[1],true_end[0]])
        if osm_ext: osm_feats.append(ofs)
    return np.array(tok,dtype=np.float32),np.array(meta,dtype=np.float32),np.array(y,dtype=np.float32),np.array(osm_feats,dtype=np.float32) if osm_ext else None

def build_grid_ds(rows, tok, osm_ext=None, label=''):
    ids,aux,meta,y,osm_f=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:SEQ_POINTS]
        if len(pts)<SEQ_POINTS: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        gi,ga,go=[],[],[]
        for i,(lon,lat) in enumerate(pts):
            gid,ox,oy=tok.encode(lon,lat); gi.append(gid+1); ga.append([ox,oy])
            if osm_ext:
                pfx=f'p{i}'; fe=osm_ext.point_features(lon,lat,pfx)
                go.append([fe[f'{pfx}_nearest_major_m']/1500.0,fe[f'{pfx}_nearest_secondary_m']/1500.0,
                           fe[f'{pfx}_local_road_count_r200']/20.0,fe[f'{pfx}_major_road_count_r300']/10.0,
                           fe[f'{pfx}_node_density_r200']/50.0])
        true_end=r['poly'][-1]; strt=haversine_km(pts[0][1],pts[0][0],pts[-1][1],pts[-1][0])*1000
        ids.append(gi); aux.append(ga); meta.append([(r['ts']//3600)%24,(r['ts']//86400)%7,strt/1000,len(pts)/float(SEQ_POINTS)])
        y.append([true_end[1],true_end[0]])
        if osm_ext: osm_f.append(go)
    return np.array(ids,dtype=np.int64),np.array(aux,dtype=np.float32),np.array(meta,dtype=np.float32),np.array(y,dtype=np.float32),np.array(osm_f,dtype=np.float32) if osm_ext else None

def build_cluster_ds(rows, tok, osm_ext=None, label=''):
    ids,aux,meta,y,osm_f=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:SEQ_POINTS]
        if len(pts)<SEQ_POINTS: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        ci,ca,co=[],[],[]
        for i,(lon,lat) in enumerate(pts):
            cid,rx,ry=tok.encode(lon,lat); ci.append(cid+1); ca.append([rx,ry])
            if osm_ext:
                pfx=f'p{i}'; fe=osm_ext.point_features(lon,lat,pfx)
                co.append([fe[f'{pfx}_nearest_major_m']/1500.0,fe[f'{pfx}_nearest_secondary_m']/1500.0,
                           fe[f'{pfx}_local_road_count_r200']/20.0,fe[f'{pfx}_major_road_count_r300']/10.0,
                           fe[f'{pfx}_node_density_r200']/50.0])
        true_end=r['poly'][-1]; strt=haversine_km(pts[0][1],pts[0][0],pts[-1][1],pts[-1][0])*1000
        ids.append(ci); aux.append(ca); meta.append([(r['ts']//3600)%24,(r['ts']//86400)%7,strt/1000,len(pts)/float(SEQ_POINTS)])
        y.append([true_end[1],true_end[0]])
        if osm_ext: osm_f.append(co)
    return np.array(ids,dtype=np.int64),np.array(aux,dtype=np.float32),np.array(meta,dtype=np.float32),np.array(y,dtype=np.float32),np.array(osm_f,dtype=np.float32) if osm_ext else None

class ContDestTransformer(nn.Module):
    def __init__(self, tok_dim, meta_dim, osm_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.tp=nn.Linear(tok_dim,d); self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*2,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,2))
        self.uosm=osm_dim>0
        if self.uosm: self.op=nn.Linear(osm_dim,d)
    def forward(self,tok,meta,osm=None):
        x=self.tp(tok)
        if self.uosm and osm is not None: x=x+self.op(osm)
        cls=self.cls.expand(tok.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,mf],1))

class TokenDestTransformer(nn.Module):
    def __init__(self, vocab, aux_dim, meta_dim, osm_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.emb=nn.Embedding(vocab+1,d,padding_idx=0); self.ap=nn.Linear(aux_dim,d)
        self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*2,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,2))
        self.uosm=osm_dim>0
        if self.uosm: self.op=nn.Linear(osm_dim,d)
    def forward(self,ids,aux,meta,osm=None):
        x=self.emb(ids)+self.ap(aux)
        if self.uosm and osm is not None: x=x+self.op(osm)
        cls=self.cls.expand(ids.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,mf],1))

def haversine_err(pred, true):
    lat1,lon1=np.radians(pred[:,0]),np.radians(pred[:,1])
    lat2,lon2=np.radians(true[:,0]),np.radians(true[:,1])
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R/1000*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_cont(Xt,Xm,Xo,y,label,osm_dim,ep=EPOCHS,bs=BS):
    n=len(Xt); idx=np.arange(n)
    tr,te=train_test_split(idx,test_size=0.2,random_state=42)
    tr,va=train_test_split(tr,test_size=0.1,random_state=42)
    ytr,yva,yte=y[tr],y[va],y[te]
    tm=Xt[tr].mean(axis=(0,1)); ts=Xt[tr].std(axis=(0,1))+1e-6
    mm=Xm[tr].mean(0); ms=Xm[tr].std(0)+1e-6
    Xt=(Xt-tm)/ts; Xm=(Xm-mm)/ms
    if Xo is not None:
        om=Xo[tr].mean(axis=(0,1)); os_=Xo[tr].std(axis=(0,1))+1e-6
        Xo=(Xo-om)/os_
    yt=torch.tensor(ytr).to(DEV); yv=torch.tensor(yva).to(DEV); yte_t=torch.tensor(yte)
    Xtr_t=torch.tensor(Xt[tr]).to(DEV); Xva_t=torch.tensor(Xt[va]).to(DEV); Xte_t=torch.tensor(Xt[te]).to(DEV)
    Xtr_m=torch.tensor(Xm[tr]).to(DEV); Xva_m=torch.tensor(Xm[va]).to(DEV); Xte_m=torch.tensor(Xm[te]).to(DEV)
    Xtr_o=torch.tensor(Xo[tr]).to(DEV) if Xo is not None else None
    Xva_o=torch.tensor(Xo[va]).to(DEV) if Xo is not None else None; Xte_o=torch.tensor(Xo[te]).to(DEV) if Xo is not None else None
    model=ContDestTransformer(Xt.shape[2],Xm.shape[1],osm_dim,Xt.shape[1]).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; o=Xtr_o[b] if Xtr_o is not None else None
            loss=crit(model(Xtr_t[b],Xtr_m[b],o),yt[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            vl=float(crit(model(Xva_t,Xva_m,Xva_o),yv).item())
        sch.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in model.state_dict().items()}; w=0
        else: w+=1
        if w>=PATIENCE: break
    if bs_: model.load_state_dict(bs_)
    model.eval()
    with torch.no_grad(): pred=model(Xte_t,Xte_m,Xte_o).cpu().numpy()
    err=haversine_err(pred,yte); me=np.mean(err); print(f"  {label}: {me:.3f} km")
    return me

def train_token(ids,aux,meta,osm,y,label,vocab,osm_dim,ep=EPOCHS,bs=BS):
    n=len(ids); idx=np.arange(n)
    tr,te=train_test_split(idx,test_size=0.2,random_state=42)
    tr,va=train_test_split(tr,test_size=0.1,random_state=42)
    ytr,yva,yte=y[tr],y[va],y[te]
    aux=(aux-aux[tr].mean(axis=(0,1)))/(aux[tr].std(axis=(0,1))+1e-6)
    meta=(meta-meta[tr].mean(0))/(meta[tr].std(0)+1e-6)
    if osm is not None:
        om=osm[tr].mean(axis=(0,1)); os_=osm[tr].std(axis=(0,1))+1e-6; osm=(osm-om)/os_
    yt=torch.tensor(ytr).to(DEV); yv=torch.tensor(yva).to(DEV); yte_t=torch.tensor(yte)
    Xtr_i=torch.tensor(ids[tr],dtype=torch.long).to(DEV); Xva_i=torch.tensor(ids[va],dtype=torch.long).to(DEV); Xte_i=torch.tensor(ids[te],dtype=torch.long).to(DEV)
    Xtr_a=torch.tensor(aux[tr]).to(DEV); Xva_a=torch.tensor(aux[va]).to(DEV); Xte_a=torch.tensor(aux[te]).to(DEV)
    Xtr_m=torch.tensor(meta[tr]).to(DEV); Xva_m=torch.tensor(meta[va]).to(DEV); Xte_m=torch.tensor(meta[te]).to(DEV)
    Xtr_o=torch.tensor(osm[tr]).to(DEV) if osm is not None else None; Xva_o=torch.tensor(osm[va]).to(DEV) if osm is not None else None; Xte_o=torch.tensor(osm[te]).to(DEV) if osm is not None else None
    model=TokenDestTransformer(vocab,aux.shape[2],meta.shape[1],osm_dim,ids.shape[1]).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); bv=float('inf'); bs_=None; w=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; o=Xtr_o[b] if Xtr_o is not None else None
            loss=crit(model(Xtr_i[b],Xtr_a[b],Xtr_m[b],o),yt[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            vl=float(crit(model(Xva_i,Xva_a,Xva_m,Xva_o),yv).item())
        sch.step(vl)
        if vl<bv-1e-4: bv=vl; bs_={k:v.clone() for k,v in model.state_dict().items()}; w=0
        else: w+=1
        if w>=PATIENCE: break
    if bs_: model.load_state_dict(bs_)
    model.eval()
    with torch.no_grad(): pred=model(Xte_i,Xte_a,Xte_m,Xte_o).cpu().numpy()
    err=haversine_err(pred,yte); me=np.mean(err); print(f"  {label}: {me:.3f} km")
    return me

def main():
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
    rows=load_clean(N_SAMPLE)
    gt=GridTokenizer(); ct=ClusterTokenizer(256)
    tp=[p for r in rows[:int(len(rows)*0.8)] for p in r['poly'][:SEQ_POINTS]]; ct.fit(tp)
    print(f"  Grid:{gt.vocab}, Cluster:256")
    ose=OSMFeatureExtractor('data/porto_highways_bbox.osm.json') if _os.path.exists('data/porto_highways_bbox.osm.json') else None

    print("\nNon-OSM datasets..."); t0=time.time()
    ct_t,ct_m,ct_y,_=build_cont_ds(rows,label='cont')
    gi,ga,gm,gy,_=build_grid_ds(rows,gt,label='grid')
    ci,ca,cm,cy,_=build_cluster_ds(rows,ct,label='clust')
    print(f"  {time.time()-t0:.0f}s  Cont:{len(ct_t)} Grid:{len(gi)} Cluster:{len(ci)}")

    orow=rows[:min(N_OSM_SAMPLE,len(rows))]
    print("\nOSM datasets..."); t0=time.time()
    _,_,_,co=build_cont_ds(orow,ose,label='cont+osm')
    _,_,_,_,go=build_grid_ds(orow,gt,ose,label='grid+osm')
    _,_,_,_,cso=build_cluster_ds(orow,ct,ose,label='clust+osm')
    print(f"  {time.time()-t0:.0f}s")
    n_osm=min(len(co),len(go),len(cso))

    res={}
    print("\n"+"="*70); print("  TASK 3: DESTINATION — 3×2 CROSS EXPERIMENT"); print("="*70)
    print("\n── Full (no OSM) ──")
    res['c_no']=train_cont(ct_t,ct_m,None,ct_y,"Continuous",0)
    res['g_no']=train_token(gi,ga,gm,None,gy,"Grid Token",gt.vocab,0)
    res['cl_no']=train_token(ci,ca,cm,None,cy,"Cluster Token",256,0)
    print("\n── Subset (no OSM vs OSM→token emb) ──")
    res['c_no_sub']=train_cont(ct_t[:n_osm],ct_m[:n_osm],None,ct_y[:n_osm],"Continuous (sub)",0)
    res['c_osm']=train_cont(ct_t[:n_osm],ct_m[:n_osm],co[:n_osm],ct_y[:n_osm],"Continuous+OSM",len(OSM_KEYS))
    res['g_no_sub']=train_token(gi[:n_osm],ga[:n_osm],gm[:n_osm],None,gy[:n_osm],"Grid Token (sub)",gt.vocab,0)
    res['g_osm']=train_token(gi[:n_osm],ga[:n_osm],gm[:n_osm],go[:n_osm],gy[:n_osm],"Grid Token+OSM",gt.vocab,len(OSM_KEYS))
    res['cl_no_sub']=train_token(ci[:n_osm],ca[:n_osm],cm[:n_osm],None,cy[:n_osm],"Cluster Token (sub)",256,0)
    res['cl_osm']=train_token(ci[:n_osm],ca[:n_osm],cm[:n_osm],cso[:n_osm],cy[:n_osm],"Cluster Token+OSM",256,len(OSM_KEYS))

    print("\n"+"="*70); print("  SUMMARY"); print("="*70)
    print("  Full dataset (no OSM):")
    for k,n in [('c_no','Continuous'),('g_no','Grid Token'),('cl_no','Cluster Token')]:
        print(f"    {n:<20s} {res[k]:.3f} km")
    print("  OSM ablation:")
    print(f"    {'Method':<22s} {'No OSM':>7s} {'+OSM':>7s} {'Δ':>7s}")
    print(f"    {'-'*45}")
    for k_no,k_os,nm in [('c_no_sub','c_osm','Continuous'),('g_no_sub','g_osm','Grid Token'),('cl_no_sub','cl_osm','Cluster Token')]:
        nv=res[k_no]; ov=res[k_os]; d=((ov-nv)/max(nv,1e-6))*100
        print(f"    {nm:<22s} {nv:>7.3f} {ov:>7.3f} {d:>+6.2f}%")

    sp='output/task3_token_osm_summary.md'; _os.makedirs('output',exist_ok=True)
    with open(sp,'w') as f:
        f.write("# Task 3: Destination Prediction — Tokenization × OSM\n\n")
        f.write(f"Config: {SEQ_POINTS}-point prefix, {N_SAMPLE} total, OSM subset={n_osm}\n\n")
        f.write("## Full dataset (no OSM)\n\n")
        for k,n in [('c_no','Continuous'),('g_no','Grid Token'),('cl_no','Cluster Token')]:
            f.write(f"- {n}: {res[k]:.3f} km\n")
        f.write("\n## OSM→token embedding\n\n")
        f.write("| Method | No OSM (km) | +OSM tok emb (km) | Δ% |\n|---|---:|---:|---:|\n")
        for k_no,k_os,nm in [('c_no_sub','c_osm','Continuous'),('g_no_sub','g_osm','Grid Token'),('cl_no_sub','cl_osm','Cluster Token')]:
            nv=res[k_no]; ov=res[k_os]; d=((ov-nv)/max(nv,1e-6))*100
            f.write(f"| {nm} | {nv:.3f} | {ov:.3f} | {d:+.2f}% |\n")
    print(f"\nSaved to {sp}")

if __name__=="__main__": main()
