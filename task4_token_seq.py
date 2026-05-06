#!/usr/bin/env python3
"""Task 4: Grid/Cluster Token SeqTransformer for duration prediction."""
import sys, os, math, json, ast
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

R = 6371000.0
def haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}
def point_in_porto(lon,lat):
    return PORTO['lon_min']<=lon<=PORTO['lon_max'] and PORTO['lat_min']<=lat<=PORTO['lat_max']

# ---- Tokenizers ----
class GridTokenizer:
    def __init__(self, cl=0.0018, co=0.0022):
        self.nl = int(math.ceil((41.3-40.8)/cl)); self.nlo = int(math.ceil((-8.4+8.8)/co))
        self.vocab = self.nl*self.nlo; self.cl,self.co=cl,co
    def encode(self,lon,lat):
        li=int(np.clip(np.floor((lat-40.8)/self.cl),0,self.nl-1))
        lj=int(np.clip(np.floor((lon+8.8)/self.co),0,self.nlo-1))
        return li*self.nlo+lj, float(np.clip((lat-40.8)/self.cl-li,0,0.999)), float(np.clip((lon+8.8)/self.co-lj,0,0.999))

class ClusterTokenizer:
    def __init__(self, n_clusters=256):
        self.n_clusters=n_clusters; self.kmeans=None; self.centers=None
    def fit(self,pts):
        pts=np.asarray(pts,dtype=np.float32)
        if len(pts)>200000: pts=pts[np.random.choice(len(pts),200000,replace=False)]
        self.kmeans=MiniBatchKMeans(n_clusters=self.n_clusters,batch_size=4096,random_state=42,n_init=5).fit(pts)
        self.centers=self.kmeans.cluster_centers_.astype(np.float32)
    def encode(self,lon,lat):
        cid=int(np.argmin(np.sum((self.centers-[lon,lat])**2,axis=1)))
        return cid, float(lon-self.centers[cid][0]), float(lat-self.centers[cid][1])

# ---- Load & Clean (same as taxi_duration_prediction.py) ----
print("Loading data...")
rows = []
with open('kaggle_dl/train.csv') as f:
    import csv
    for i, row in enumerate(csv.DictReader(f)):
        if row['MISSING_DATA']=='True': continue
        try: pts=ast.literal_eval(row['POLYLINE'])
        except: continue
        if len(pts)<10: continue
        if any(not point_in_porto(p[0],p[1]) for p in pts): continue
        # Simplified spike removal
        for _ in range(2):
            speeds=[]
            for j in range(1,len(pts)):
                speeds.append(haversine(pts[j-1][1],pts[j-1][0],pts[j][1],pts[j][0])/15*3.6)
            to_rm=[]
            for j in range(1,len(pts)-1):
                if speeds[j-1]>120 and speeds[j]>120:
                    ac=haversine(pts[j-1][1],pts[j-1][0],pts[j+1][1],pts[j+1][0])/15*3.6
                    if ac<84: to_rm.append(j)
            for j in reversed(to_rm): pts.pop(j)
            if not to_rm: break
        speeds=[haversine(pts[j-1][1],pts[j-1][0],pts[j][1],pts[j][0])/15*3.6 for j in range(1,len(pts))]
        if max(speeds)>300: continue
        if np.mean([s>120 for s in speeds])>0.15: continue
        dur=(len(pts)-1)*15
        if dur<60 or dur>7200: continue
        rows.append({'poly':pts,'ts':int(row['TIMESTAMP']),'dur':float(dur),'call':row.get('CALL_TYPE','C')})
        if len(rows)>=25000: break
print(f"Cleaned: {len(rows)} records")

# ---- Build datasets ----
SEQ_POINTS=20  # smaller for speed
def build_meta(prefix_pts, ts, call):
    hour=(ts//3600)%24; wday=(ts//86400)%7
    slon,slat=prefix_pts[0]; elon,elat=prefix_pts[-1]
    seg=[haversine(prefix_pts[i-1][1],prefix_pts[i-1][0],prefix_pts[i][1],prefix_pts[i][0]) for i in range(1,len(prefix_pts))]
    td=sum(seg); straight=haversine(slat,slon,elat,elon)
    detour=td/max(1,straight)
    speeds=[s/15 for s in seg]
    avg_speed=np.mean(speeds) if speeds else 0
    recent3=np.mean(speeds[-3:]) if len(speeds)>=3 else avg_speed
    return np.array([hour,wday,int(hour in{8,9,17,18,19}),call_enc(call),float(len(prefix_pts)),
                     td,straight,detour,avg_speed,recent3],dtype=np.float32)
def call_enc(c): return {'A':0,'B':1,'C':2}.get(c,2)

print("Building continuous tokens...")
cont_tokens, cont_meta, cont_y = [], [], []
for r in rows:
    pts=r['poly'][:SEQ_POINTS]
    if len(pts)<SEQ_POINTS: continue
    slon,slat=pts[0]; prev_lon,prev_lat=slon,slat; cum=0; prev_b=0
    toks=[]
    for i,(lon,lat) in enumerate(pts):
        dx=(lon-slon)*111320*math.cos(math.radians(41.15))
        dy=(lat-slat)*111320
        if i==0: step_x=step_y=sd=sp=0; tb=0
        else:
            step_x=(lon-prev_lon)*111320*math.cos(math.radians(41.15))
            step_y=(lat-prev_lat)*111320
            sd=haversine(prev_lat,prev_lon,lat,lon); sp=sd/15
            c_b=math.atan2(lat-prev_lat,lon-prev_lon)
            tb=((c_b-prev_b+math.pi)%(2*math.pi)-math.pi)/math.pi
            prev_b=c_b; cum+=sd
        toks.append([dx/1000,dy/1000,step_x/1000,step_y/1000,sd/1000,sp/10,tb,cum/1000,i/max(1,SEQ_POINTS-1)])
        prev_lon,prev_lat=lon,lat
    cont_tokens.append(toks); cont_meta.append(build_meta(pts,r['ts'],r['call'])); cont_y.append(r['dur'])
cont_tokens=np.array(cont_tokens,dtype=np.float32); cont_meta=np.array(cont_meta,dtype=np.float32); cont_y=np.array(cont_y,dtype=np.float32)
print(f"  {len(cont_tokens)} samples, token_dim={cont_tokens.shape[2]}, meta_dim={cont_meta.shape[1]}")

print("Building Grid tokens...")
grid_tok=GridTokenizer()
grid_ids,grid_aux,grid_meta,grid_y=[],[],[],[]
for r in rows:
    pts=r['poly'][:SEQ_POINTS]
    if len(pts)<SEQ_POINTS: continue
    gi,ga=[],[]
    for lon,lat in pts:
        gid,ox,oy=grid_tok.encode(lon,lat); gi.append(gid+1); ga.append([ox,oy])
    grid_ids.append(gi); grid_aux.append(ga); grid_meta.append(build_meta(pts,r['ts'],r['call'])); grid_y.append(r['dur'])
grid_ids=np.array(grid_ids,dtype=np.int64); grid_aux=np.array(grid_aux,dtype=np.float32)
grid_meta=np.array(grid_meta,dtype=np.float32); grid_y=np.array(grid_y,dtype=np.float32)
print(f"  {len(grid_ids)} samples, grid_vocab={grid_tok.vocab}")

print("Building Cluster tokens...")
cluster_tok=ClusterTokenizer(256)
train_pts=[p for r in rows[:int(len(rows)*0.8)] for p in r['poly'][:SEQ_POINTS]]
cluster_tok.fit(train_pts)
cluster_ids,cluster_aux,cluster_meta,cluster_y=[],[],[],[]
for r in rows:
    pts=r['poly'][:SEQ_POINTS]
    if len(pts)<SEQ_POINTS: continue
    ci,ca=[],[]
    for lon,lat in pts:
        cid,rx,ry=cluster_tok.encode(lon,lat); ci.append(cid+1); ca.append([rx,ry])
    cluster_ids.append(ci); cluster_aux.append(ca); cluster_meta.append(build_meta(pts,r['ts'],r['call'])); cluster_y.append(r['dur'])
cluster_ids=np.array(cluster_ids,dtype=np.int64); cluster_aux=np.array(cluster_aux,dtype=np.float32)
cluster_meta=np.array(cluster_meta,dtype=np.float32); cluster_y=np.array(cluster_y,dtype=np.float32)
print(f"  {len(cluster_ids)} samples, cluster_vocab=256")

# ---- SeqTransformer models ----
class ContSeqTransformer(nn.Module):
    def __init__(self, tok_dim, meta_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.tp=nn.Linear(tok_dim,d); self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(self,tok,meta):
        x=self.tp(tok); cls=self.cls.expand(tok.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,pool,mf],1)).squeeze(1)

class TokenSeqTransformer(nn.Module):
    def __init__(self, vocab, aux_dim, meta_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.emb=nn.Embedding(vocab+1,d,padding_idx=0); self.ap=nn.Linear(aux_dim,d)
        self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
    def forward(self,ids,aux,meta):
        x=self.emb(ids)+self.ap(aux); cls=self.cls.expand(ids.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,pool,mf],1)).squeeze(1)

# ---- Training ----
def train_cont(Xt,Xm,y,label,ep=20,bs=256):
    n=len(Xt); idx=np.arange(n)
    tr_i,te_i=train_test_split(idx,test_size=0.2,random_state=42)
    tr_i,va_i=train_test_split(tr_i,test_size=0.1,random_state=42)
    y_log=np.log1p(y); ytr, yva, yte=y_log[tr_i],y_log[va_i],y_log[te_i]
    Xt=(Xt-Xt[tr_i].mean(axis=(0,1)))/(Xt[tr_i].std(axis=(0,1))+1e-6)
    Xm=(Xm-Xm[tr_i].mean(0))/(Xm[tr_i].std(0)+1e-6)
    Xtr_t=torch.tensor(Xt[tr_i]); Xva_t=torch.tensor(Xt[va_i]); Xte_t=torch.tensor(Xt[te_i])
    Xtr_m=torch.tensor(Xm[tr_i]); Xva_m=torch.tensor(Xm[va_i]); Xte_m=torch.tensor(Xm[te_i])
    ytr_t=torch.tensor(ytr); yva_t=torch.tensor(yva); yte_t=torch.tensor(yte)
    model=ContSeqTransformer(Xt.shape[2],Xm.shape[1],Xt.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); best_va=float('inf'); best_st=None; wait=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; loss=crit(model(Xtr_t[b],Xtr_m[b]),ytr_t[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            va_loss=float(crit(model(Xva_t,Xva_m),yva_t).item())
        sched.step(va_loss)
        if va_loss<best_va-1e-4: best_va=va_loss; best_st={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=4: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad(): pred=np.expm1(model(Xte_t,Xte_m).cpu().numpy())
    true=np.expm1(yte_t.cpu().numpy()); mae=np.mean(np.abs(pred-true)); rmse=np.sqrt(np.mean((pred-true)**2))
    print(f"  {label}: MAE={mae:.1f}s RMSE={rmse:.1f}s")
    return mae

def train_token(ids_t,aux_t,meta_t,y_t,label,vocab,ep=20,bs=256):
    n=len(ids_t); idx=np.arange(n)
    tr_i,te_i=train_test_split(idx,test_size=0.2,random_state=42)
    tr_i,va_i=train_test_split(tr_i,test_size=0.1,random_state=42)
    y_log=np.log1p(y_t); ytr, yva, yte=y_log[tr_i],y_log[va_i],y_log[te_i]
    aux_t=(aux_t-aux_t[tr_i].mean(axis=(0,1)))/(aux_t[tr_i].std(axis=(0,1))+1e-6)
    meta_t=(meta_t-meta_t[tr_i].mean(0))/(meta_t[tr_i].std(0)+1e-6)
    Xtr_i=torch.tensor(ids_t[tr_i],dtype=torch.long); Xva_i=torch.tensor(ids_t[va_i],dtype=torch.long); Xte_i=torch.tensor(ids_t[te_i],dtype=torch.long)
    Xtr_a=torch.tensor(aux_t[tr_i]); Xva_a=torch.tensor(aux_t[va_i]); Xte_a=torch.tensor(aux_t[te_i])
    Xtr_m=torch.tensor(meta_t[tr_i]); Xva_m=torch.tensor(meta_t[va_i]); Xte_m=torch.tensor(meta_t[te_i])
    ytr_t=torch.tensor(ytr); yva_t=torch.tensor(yva); yte_t=torch.tensor(yte)
    model=TokenSeqTransformer(vocab,aux_t.shape[2],meta_t.shape[1],ids_t.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); best_va=float('inf'); best_st=None; wait=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; loss=crit(model(Xtr_i[b],Xtr_a[b],Xtr_m[b]),ytr_t[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            va_loss=float(crit(model(Xva_i,Xva_a,Xva_m),yva_t).item())
        sched.step(va_loss)
        if va_loss<best_va-1e-4: best_va=va_loss; best_st={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=4: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad(): pred=np.expm1(model(Xte_i,Xte_a,Xte_m).cpu().numpy())
    true=np.expm1(yte_t.cpu().numpy()); mae=np.mean(np.abs(pred-true))
    print(f"  {label}: MAE={mae:.1f}s")
    return mae

# ---- Run all ----
print("\nTraining...")
c_mae=train_cont(cont_tokens,cont_meta,cont_y,"Continuous SeqTransformer",ep=20)
g_mae=train_token(grid_ids,grid_aux,grid_meta,grid_y,"Grid Token SeqTransformer",grid_tok.vocab,ep=20)
cl_mae=train_token(cluster_ids,cluster_aux,cluster_meta,cluster_y,"Cluster Token SeqTransformer",256,ep=20)

print(f"\n{'='*50}")
print(f"  TASK 4: TOKEN SeqTransformer COMPARISON")
print(f"  {'Method':<30s} {'MAE (s)':>8s}")
print(f"  {'-'*40}")
print(f"  {'Continuous SeqTransformer':<30s} {c_mae:8.1f}")
print(f"  {'Grid Token SeqTransformer':<30s} {g_mae:8.1f}")
print(f"  {'Cluster Token SeqTransformer':<30s} {cl_mae:8.1f}")
print(f"{'='*50}")
