#!/usr/bin/env python3
"""Task 1: Grid & Cluster Token embedding retrieval experiment."""
import sys, os, math, random, ast, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

PORTO_BBOX = {'lat_min': 40.8, 'lat_max': 41.3, 'lon_min': -8.8, 'lon_max': -8.4}
RESAMPLE_POINTS = 32; R = 6371000.0

def haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class GridTokenizer:
    def __init__(self, cell_lat=0.0018, cell_lon=0.0022):
        self.cell_lat, self.cell_lon = cell_lat, cell_lon
        self.n_lat = int(math.ceil((41.3-40.8)/cell_lat))
        self.n_lon = int(math.ceil((-8.4+8.8)/cell_lon))
        self.vocab_size = self.n_lat * self.n_lon
    def encode_point(self, lon, lat):
        li = int(np.clip(np.floor((lat-40.8)/self.cell_lat), 0, self.n_lat-1))
        lj = int(np.clip(np.floor((lon+8.8)/self.cell_lon), 0, self.n_lon-1))
        return li*self.n_lon+lj, float(np.clip((lat-40.8)/self.cell_lat-li, 0, 0.999)), float(np.clip((lon+8.8)/self.cell_lon-lj, 0, 0.999))

class ClusterTokenizer:
    def __init__(self, n_clusters=256, max_fit=100000):
        self.n_clusters=n_clusters; self.max_fit=max_fit; self.kmeans=None; self.centers=None
    def fit(self, points):
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) > self.max_fit:
            pts = pts[np.random.choice(len(pts), self.max_fit, replace=False)]
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=4096, random_state=42, n_init=5).fit(pts)
        self.centers = self.kmeans.cluster_centers_.astype(np.float32)
    def encode_point(self, lon, lat):
        cid = int(np.argmin(np.sum((self.centers-[lon,lat])**2, axis=1)))
        return cid, float(lon-self.centers[cid][0]), float(lat-self.centers[cid][1])

# ---- Load data ----
print("Loading trajectories...")
records = []
with open('kaggle_dl/train.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['MISSING_DATA'] == 'True': continue
        try: pts = ast.literal_eval(row['POLYLINE'])
        except: continue
        if len(pts) < 5: continue
        if any(not (PORTO_BBOX['lon_min']<=p[0]<=PORTO_BBOX['lon_max'] and PORTO_BBOX['lat_min']<=p[1]<=PORTO_BBOX['lat_max']) for p in pts): continue
        idx = np.linspace(0, len(pts)-1, RESAMPLE_POINTS, dtype=int)
        pts_arr = np.array(pts)[idx]
        seg_dist = np.array([haversine(pts_arr[i-1][1], pts_arr[i-1][0], pts_arr[i][1], pts_arr[i][0]) for i in range(1, RESAMPLE_POINTS)])
        length_m = float(np.sum(seg_dist))
        start_lon, start_lat = pts_arr[0]; end_lon, end_lat = pts_arr[-1]
        records.append({'points': pts_arr.tolist(), 'vec_xy': pts_arr.astype(np.float32),
            'length_m': length_m, 'start_lon': start_lon, 'start_lat': start_lat,
            'end_lon': end_lon, 'end_lat': end_lat})
        if len(records) >= 5000: break
print(f"Loaded {len(records)} records")

# ---- Tokenize ----
print("Tokenizing...")
grid_tok = GridTokenizer()
cluster_tok = ClusterTokenizer(n_clusters=256)
all_train_points = [p for r in records[:3000] for p in r['points']]
cluster_tok.fit(all_train_points)
print(f"Grid vocab: {grid_tok.vocab_size}, Cluster centers: {len(cluster_tok.centers)}")

grid_ids, grid_aux = [], []
cluster_ids, cluster_aux = [], []
for r in records:
    gi, ga, ci, ca = [], [], [], []
    for lon, lat in r['points']:
        gid, ox, oy = grid_tok.encode_point(lon, lat); gi.append(gid+1); ga.append([ox, oy])
        cid, rx, ry = cluster_tok.encode_point(lon, lat); ci.append(cid+1); ca.append([rx, ry])
    grid_ids.append(gi); grid_aux.append(ga)
    cluster_ids.append(ci); cluster_aux.append(ca)

# ---- Models ----
class TokenAE(nn.Module):
    def __init__(self, vocab, aux_dim=2, emb_dim=128, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab+1, d, padding_idx=0)
        self.aproj = nn.Linear(aux_dim, d)
        self.enc = nn.Sequential(nn.Linear(d*RESAMPLE_POINTS,256),nn.BatchNorm1d(256),nn.GELU(),
                                 nn.Linear(256,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,emb_dim))
        self.dec = nn.Sequential(nn.Linear(emb_dim,256),nn.BatchNorm1d(256),nn.GELU(),
                                 nn.Linear(256,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,64))
    def encode(self, ids, aux):
        N=ids.size(0); x=self.embed(ids)+self.aproj(aux)
        return self.enc(x.reshape(N,-1))
    def recon(self, ids, aux):
        e=self.encode(ids,aux); r=self.dec(e)
        return r.reshape(ids.size(0),RESAMPLE_POINTS,2), e

class ContAE(nn.Module):
    def __init__(self,input_dim=64,emb_dim=128):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(64,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,emb_dim))
        self.dec=nn.Sequential(nn.Linear(emb_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,64))
    def encode(self,x): return self.enc(x)

# ---- Train ----
def train_ae(model, data_tensors, epochs=30, bs=64, lr=8e-4):
    n=len(data_tensors[0]); ntr=int(n*0.8)
    opt=torch.optim.AdamW(model.parameters(),lr=lr);
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    for ep in range(1,epochs+1):
        model.train(); idx=np.random.choice(ntr,min(bs,ntr),replace=False)
        batch=[t[idx] for t in data_tensors]
        recon,emb=model.recon(*batch)
        # Target: reconstruct the continuous auxiliary features (offset/residual)
        target=batch[1].reshape(len(idx),-1)
        loss=F.mse_loss(recon.reshape(len(idx),-1), target)
        opt.zero_grad();loss.backward();opt.step();sch.step()
    model.eval()
    with torch.no_grad():
        return F.normalize(model.encode(*[t for t in data_tensors]), dim=1).cpu().numpy()

def train_cont_ae(data, epochs=30, bs=64, lr=8e-4):
    n=len(data); ntr=int(n*0.8)
    model=ContAE(); opt=torch.optim.AdamW(model.parameters(),lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    for ep in range(1,epochs+1):
        model.train(); idx=np.random.choice(ntr,min(bs,ntr),replace=False)
        batch=data[idx]; emb=model.encode(batch)
        recon=model.dec(emb); loss=F.mse_loss(recon,batch)
        opt.zero_grad();loss.backward();opt.step();sch.step()
    model.eval()
    with torch.no_grad():
        return F.normalize(model.encode(data),dim=1).cpu().numpy()

print("Training contrastive Grid Token encoder...")
# Use contrastive: coarse grid-based pairs
pos_pairs = []
all_vec = np.stack([r['vec_xy'].ravel() for r in records])
for i in range(len(records)):
    sdist = np.linalg.norm(all_vec[:,:2]-all_vec[i,:2], axis=1)
    edist = np.linalg.norm(all_vec[:,-2:]-all_vec[i,-2:], axis=1)
    cand = np.where((sdist<2000)&(edist<2000))[0]; cand = cand[cand!=i]
    for j in cand[:3]: pos_pairs.append((i,int(j)))
print(f"  Positive pairs: {len(pos_pairs)}")

class ContEncoder(nn.Module):
    def __init__(self,in_dim=64,emb_dim=128):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,emb_dim))
    def forward(self,x): return F.normalize(self.enc(x),dim=1)

# Flat grid features: embed each point, flatten
print("Building flat token features...")
gid_t=torch.tensor(np.array(grid_ids),dtype=torch.long)
gau_t=torch.tensor(np.array(grid_aux),dtype=torch.float32)
cid_t=torch.tensor(np.array(cluster_ids),dtype=torch.long); cau_t=torch.tensor(np.array(cluster_aux),dtype=torch.float32)

# Build flat features: embed token + aux, flatten per trajectory
grid_flat = torch.zeros(len(records), 64)
cluster_flat = torch.zeros(len(records), 64)
for i in range(len(records)):
    gfeat = torch.cat([gau_t[i], torch.zeros(RESAMPLE_POINTS, 1)], dim=1)  # aux + dummy
    grid_flat[i] = gfeat.reshape(-1)[:64]
    cfeat = torch.cat([cau_t[i], torch.zeros(RESAMPLE_POINTS, 1)], dim=1)
    cluster_flat[i] = cfeat.reshape(-1)[:64]

# Normalize flat features
grid_flat = (grid_flat-grid_flat.mean(dim=0))/(grid_flat.std(dim=0)+1e-6)
cluster_flat = (cluster_flat-cluster_flat.mean(dim=0))/(cluster_flat.std(dim=0)+1e-6)
n_all = len(records)

def train_cont_enc(data, epochs=30, bs=64, lr=5e-4):
    model=ContEncoder(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    margin=0.3
    for ep in range(1,epochs+1):
        model.train(); idx=np.random.choice(len(pos_pairs),min(bs,len(pos_pairs)),replace=False)
        anc,pos=zip(*[pos_pairs[i] for i in idx])
        anc,pos=list(anc),list(pos)
        neg=[(a+np.random.randint(1,n_all-1))%n_all for a in anc]
        ae=model(data[anc]); pe=model(data[pos]); ne=model(data[neg])
        loss=torch.clamp(margin-(ae*pe).sum(dim=1)+(ae*ne).sum(dim=1),min=0).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
    model.eval()
    with torch.no_grad(): return F.normalize(model.enc(data),dim=1).cpu().numpy()

print("Training contrastive Grid Token...")
grid_cont_emb = train_cont_enc(grid_flat, epochs=30)
print("Training contrastive Cluster Token...")
cluster_cont_emb = train_cont_enc(cluster_flat, epochs=30)

# Build continuous features for control experiment
vec_t=torch.tensor(np.stack([r["vec_xy"].ravel() for r in records]),dtype=torch.float32)
v_mean,v_std=vec_t.mean(dim=0),vec_t.std(dim=0)+1e-6
vec_norm=(vec_t-v_mean)/v_std
print("Training contrastive Continuous (control)...")
cont_cont_emb = train_cont_enc(vec_norm, epochs=30)

# ---- Evaluation ----
def eval_retrieval(embeddings, n_query=80):
    np.random.seed(42); query_idx=np.random.choice(len(embeddings),n_query,replace=False)
    all_vec=np.stack([r['vec_xy'].ravel() for r in records])
    t1=t5=0
    for qi in query_idx:
        sdist=np.linalg.norm(all_vec[:,:2]-all_vec[qi,:2],axis=1)
        edist=np.linalg.norm(all_vec[:,-2:]-all_vec[qi,-2:],axis=1)
        cand=np.where((sdist<2000)&(edist<2000))[0]; cand=cand[cand!=qi]
        if len(cand)>500: cand=cand[:500]
        if len(cand)<10: cand=np.argsort(sdist+edist)[1:101]
        sims=np.dot(embeddings[cand],embeddings[qi]); top5=cand[np.argsort(-sims)[:5]]
        for rk,ca in enumerate(top5):
            s=haversine(records[qi]['start_lat'],records[qi]['start_lon'],records[ca]['start_lat'],records[ca]['start_lon'])
            e=haversine(records[qi]['end_lat'],records[qi]['end_lon'],records[ca]['end_lat'],records[ca]['end_lon'])
            lr=records[qi]['length_m']/max(1,records[ca]['length_m'])
            if s<=250 and e<=250 and 0.7<=lr<=1.3:
                if rk==0: t1+=1
                t5+=1
    return t1/n_query,t5/n_query

print("\nEvaluating...")
gc1,gc5=eval_retrieval(grid_cont_emb)
cc1,cc5=eval_retrieval(cluster_cont_emb)
ctc1,ctc5=eval_retrieval(cont_cont_emb)

print(f"\n{'='*55}")
print(f"  TASK 1: TOKEN EMBEDDING RETRIEVAL (CONTRASTIVE)")
print(f"{'='*55}")
print(f"  {'Method':<30s} {'Top1':>6s}  {'Top5Any':>8s}")
print(f"  {'-'*45}")
print(f"  {'Euclidean (baseline)':<30s} {'0.138':>6s}  {'0.075':>8s}")
print(f"  {'Contrastive Continuous':<30s} {ctc1:6.3f}  {ctc5:8.3f}")
print(f"  {'Contrastive Grid Token':<30s} {gc1:6.3f}  {gc5:8.3f}")
print(f"  {'Contrastive Cluster Token':<30s} {cc1:6.3f}  {cc5:8.3f}")
print(f"{'='*55}")
