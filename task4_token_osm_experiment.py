#!/usr/bin/env python3
"""
Task 4: Cross experiment — 3 tokenization types × 2 OSM modes.

Tokenization types: Continuous, Grid Token + offset, Cluster Token + residual
OSM modes: No OSM, OSM injected into token embeddings (not just meta)

Results used to correct the course design report.
"""
import sys, os, math, ast, csv, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

# ── Config ──────────────────────────────────────────────────────────
PORTO = {'lat_min':40.8,'lat_max':41.3,'lon_min':-8.8,'lon_max':-8.4}
R = 6371000.0
SEQ_POINTS = 20
N_SAMPLE = 15000       
N_OSM_SAMPLE = 15000   # full dataset for OSM experiments too
EPOCHS = 18
BATCH_SIZE = 256
PATIENCE = 4
# OSM features per point (matching OSMFeatureExtractor.point_features keys)
OSM_FEAT_KEYS = ['nearest_major_m','nearest_secondary_m','local_road_count_r200',
                 'major_road_count_r300','node_density_r200']
OSM_DIM = len(OSM_FEAT_KEYS)

def point_in_porto(lon,lat):
    return PORTO['lon_min']<=lon<=PORTO['lon_max'] and PORTO['lat_min']<=lat<=PORTO['lat_max']

def haversine(lat1,lon1,lat2,lon2):
    dlat=math.radians(lat2-lat1); dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def lonlat_to_local_xy(lon,lat):
    x=lon*111320*math.cos(math.radians(41.15)); y=lat*111320
    return x,y

# ── Tokenizers ──────────────────────────────────────────────────────
class GridTokenizer:
    def __init__(self, cl=0.0018, co=0.0022):
        self.nl=int(math.ceil((41.3-40.8)/cl)); self.nlo=int(math.ceil((-8.4+8.8)/co))
        self.vocab=self.nl*self.nlo; self.cl,self.co=cl,co
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

# ── OSM Feature Extractor (imported logic, inlined for self-containment) ──
class OSMFeatureExtractor:
    ROAD_FLAGS = {'major':1,'secondary':2,'tertiary':4,'local':8,'nonmotorized':16}
    DOMINANT_ORDER = ['major','secondary','tertiary','local','nonmotorized']
    DOMINANT_ENC = {n:i+1 for i,n in enumerate(DOMINANT_ORDER)}
    CELL_SIZE_M = 120.0
    def __init__(self, osm_json_path):
        self.osm_json_path=osm_json_path; self.query_cache={}; self._load()
    @classmethod
    def map_highway_class(cls, hw):
        if not hw: return None
        if hw in {'motorway','motorway_link','trunk','trunk_link','primary','primary_link'}: return 'major'
        if hw in {'secondary','secondary_link'}: return 'secondary'
        if hw in {'tertiary','tertiary_link'}: return 'tertiary'
        if hw in {'residential','service','living_street','unclassified'}: return 'local'
        if hw in {'pedestrian','footway','cycleway','path','track','steps'}: return 'nonmotorized'
        return None
    def _cell_key(self,x,y): return int(x//self.CELL_SIZE_M), int(y//self.CELL_SIZE_M)
    def _neighbor_candidate_indices(self,x,y,radius_m):
        hop=max(1,int(math.ceil(radius_m/self.CELL_SIZE_M)))
        cx,cy=self._cell_key(x,y); out=[]
        for dx in range(-hop,hop+1):
            for dy in range(-hop,hop+1): out.extend(self.grid.get((cx+dx,cy+dy),[]))
        if not out: return np.empty(0,dtype=np.int32)
        return np.asarray(out,dtype=np.int32)
    def _nearest_distance_for_flag(self,x,y,flag_name,search_radius_m=1500.0):
        idx=self._neighbor_candidate_indices(x,y,search_radius_m)
        if len(idx)==0: return search_radius_m
        mask=(self.node_masks[idx]&self.ROAD_FLAGS[flag_name])!=0
        if not np.any(mask): return search_radius_m
        dx=self.node_x[idx[mask]]-x; dy=self.node_y[idx[mask]]-y
        return float(np.sqrt(np.min(dx*dx+dy*dy)))
    def point_features(self,lon,lat,prefix):
        cache_key=(prefix,round(lon,5),round(lat,5))
        if cache_key in self.query_cache: return self.query_cache[cache_key]
        x,y=lonlat_to_local_xy(lon,lat)
        idx_200=self._neighbor_candidate_indices(x,y,200.0)
        idx_300=self._neighbor_candidate_indices(x,y,300.0)
        def cnt(indices,radius_m,flag_name=None):
            if len(indices)==0: return 0
            dx=self.node_x[indices]-x; dy=self.node_y[indices]-y
            inside=(dx*dx+dy*dy)<=radius_m*radius_m
            if flag_name is not None: inside&=(self.node_masks[indices]&self.ROAD_FLAGS[flag_name])!=0
            return int(np.sum(inside))
        dom_counts={}
        for name in self.DOMINANT_ORDER: dom_counts[name]=cnt(idx_200,200.0,name)
        dominant_type=max(self.DOMINANT_ORDER,key=lambda n:(dom_counts[n],-self.DOMINANT_ENC[n]))
        dom_enc=self.DOMINANT_ENC[dominant_type] if dom_counts[dominant_type]>0 else 0
        features={
            f'{prefix}_nearest_major_m':self._nearest_distance_for_flag(x,y,'major'),
            f'{prefix}_nearest_secondary_m':self._nearest_distance_for_flag(x,y,'secondary'),
            f'{prefix}_local_road_count_r200':cnt(idx_200,200.0,'local'),
            f'{prefix}_major_road_count_r300':cnt(idx_300,300.0,'major'),
            f'{prefix}_node_density_r200':cnt(idx_200,200.0),
            f'{prefix}_dominant_road_type_enc':dom_enc,
        }
        self.query_cache[cache_key]=features
        return features
    def _load(self):
        print(f"  Loading OSM: {self.osm_json_path}")
        with open(self.osm_json_path,'r',encoding='utf-8') as f: data=json.load(f)
        node_coords={}; node_masks={}
        way_count=0
        for el in data.get('elements',[]):
            if el.get('type')=='node':
                lon=el.get('lon'); lat=el.get('lat')
                if lon is None or lat is None: continue
                if point_in_porto(lon,lat): node_coords[el['id']]=(float(lon),float(lat))
        for el in data.get('elements',[]):
            if el.get('type')!='way': continue
            rc=self.map_highway_class((el.get('tags') or {}).get('highway'))
            if rc is None: continue
            flag=self.ROAD_FLAGS[rc]
            valid=0
            for nid in el.get('nodes',[]):
                if nid in node_coords:
                    node_masks.setdefault(nid,0); node_masks[nid]|=flag; valid+=1
            if valid>=2: way_count+=1
        kept=[nid for nid,mask in node_masks.items() if mask>0 and nid in node_coords]
        xs,ys,masks=[],[],[]; grid={}
        for idx,nid in enumerate(kept):
            lon,lat=node_coords[nid]; x,y=lonlat_to_local_xy(lon,lat)
            xs.append(x); ys.append(y); masks.append(node_masks[nid])
            grid.setdefault(self._cell_key(x,y),[]).append(idx)
        self.node_x=np.asarray(xs,dtype=np.float32); self.node_y=np.asarray(ys,dtype=np.float32)
        self.node_masks=np.asarray(masks,dtype=np.int32); self.grid=grid
        self.node_count=len(kept); self.way_count=way_count
        print(f"  OSM ready: {self.node_count} nodes, {self.way_count} ways")

# ── Data Loading & Cleaning ─────────────────────────────────────────
def load_and_clean(n_sample):
    print(f"Loading {n_sample} records...")
    rows=[]
    with open('kaggle_dl/train.csv') as f:
        for i,row in enumerate(csv.DictReader(f)):
            if row['MISSING_DATA']=='True': continue
            try: pts=ast.literal_eval(row['POLYLINE'])
            except: continue
            if len(pts)<10: continue
            if any(not point_in_porto(p[0],p[1]) for p in pts): continue
            # spike removal (same as task4_token_seq.py)
            for _ in range(2):
                speeds=[]
                for j in range(1,len(pts)): speeds.append(haversine(pts[j-1][1],pts[j-1][0],pts[j][1],pts[j][0])/15*3.6)
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
            if len(rows)>=n_sample: break
    print(f"  Cleaned: {len(rows)} records")
    return rows

# ── Build Continuous Tokens ─────────────────────────────────────────
CONT_DIM = 13  # standard 13-dim continuous token (same as taxi_duration_prediction.py)

def call_enc(c): return {'A':0,'B':1,'C':2}.get(c,2)

def build_meta(prefix_pts, ts, call):
    hour=(ts//3600)%24; wday=(ts//86400)%7
    slon,slat=prefix_pts[0]; elon,elat=prefix_pts[-1]
    seg=[haversine(prefix_pts[i-1][1],prefix_pts[i-1][0],prefix_pts[i][1],prefix_pts[i][0]) for i in range(1,len(prefix_pts))]
    td=sum(seg); straight=haversine(slat,slon,elat,elon)
    detour=td/max(1,straight)
    speeds=[s/15 for s in seg]
    avg_speed=np.mean(speeds) if speeds else 0
    recent3=np.mean(speeds[-3:]) if len(speeds)>=3 else avg_speed
    call_enc={'A':0,'B':1,'C':2}.get(call,2)
    return np.array([hour,wday,int(hour in{8,9,17,18,19}),call_enc,float(len(prefix_pts)),
                     td,straight,detour,avg_speed,recent3],dtype=np.float32)

def build_continuous_dataset(rows, seq_points=SEQ_POINTS, osm_extractor=None, label=''):
    tokens, meta, osm_feats, y = [],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:seq_points]
        if len(pts)<seq_points: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        slon,slat=pts[0]; prev_lon,prev_lat=slon,slat; cum=0; prev_b=0; toks=[]
        osm_toks = []
        for i,(lon,lat) in enumerate(pts):
            dx=(lon-slon)*111320*math.cos(math.radians(41.15)); dy=(lat-slat)*111320
            if i==0:
                step_x=step_y=sd=sp=0; tb=0
            else:
                step_x=(lon-prev_lon)*111320*math.cos(math.radians(41.15))
                step_y=(lat-prev_lat)*111320
                sd=haversine(prev_lat,prev_lon,lat,lon); sp=sd/15
                c_b=math.atan2(lat-prev_lat,lon-prev_lon)
                tb=((c_b-prev_b+math.pi)%(2*math.pi)-math.pi)/math.pi
                prev_b=c_b; cum+=sd
            toks.append([dx/1000,dy/1000,step_x/1000,step_y/1000,sd/1000,sp/10,tb,cum/1000,i/max(1,seq_points-1)])
            bearing_rad = math.atan2(dy,dx) if i>0 else 0
            toks[-1].extend([math.sin(bearing_rad),math.cos(bearing_rad),cum/1000,sd/1000])
            prev_lon,prev_lat=lon,lat
            if osm_extractor is not None:
                pfx=f'p{i}'
                of=osm_extractor.point_features(lon,lat,pfx)
                osm_toks.append([of[f'{pfx}_nearest_major_m']/1500.0, of[f'{pfx}_nearest_secondary_m']/1500.0,
                                 of[f'{pfx}_local_road_count_r200']/20.0, of[f'{pfx}_major_road_count_r300']/10.0,
                                 of[f'{pfx}_node_density_r200']/50.0])
        tokens.append(toks); meta.append(build_meta(pts,r['ts'],r['call'])); y.append(r['dur'])
        if osm_extractor is not None: osm_feats.append(osm_toks)
    tokens=np.array(tokens,dtype=np.float32); meta=np.array(meta,dtype=np.float32)
    y=np.array(y,dtype=np.float32)
    osm_arr = np.array(osm_feats,dtype=np.float32) if osm_extractor is not None else None
    return tokens, meta, osm_arr, y

def build_grid_dataset(rows, grid_tok, seq_points=SEQ_POINTS, osm_extractor=None, label=''):
    ids,aux,meta,osm_feats,y=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:seq_points]
        if len(pts)<seq_points: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        gi,ga,go=[],[],[]
        for i,(lon,lat) in enumerate(pts):
            gid,ox,oy=grid_tok.encode(lon,lat); gi.append(gid+1); ga.append([ox,oy])
            if osm_extractor is not None:
                pfx=f'p{i}'
                of=osm_extractor.point_features(lon,lat,pfx)
                go.append([of[f'{pfx}_nearest_major_m']/1500.0, of[f'{pfx}_nearest_secondary_m']/1500.0,
                           of[f'{pfx}_local_road_count_r200']/20.0, of[f'{pfx}_major_road_count_r300']/10.0,
                           of[f'{pfx}_node_density_r200']/50.0])
        ids.append(gi); aux.append(ga); meta.append(build_meta(pts,r['ts'],r['call'])); y.append(r['dur'])
        if osm_extractor is not None: osm_feats.append(go)
    ids=np.array(ids,dtype=np.int64); aux=np.array(aux,dtype=np.float32)
    meta=np.array(meta,dtype=np.float32); y=np.array(y,dtype=np.float32)
    osm_arr = np.array(osm_feats,dtype=np.float32) if osm_extractor is not None else None
    return ids, aux, meta, osm_arr, y

def build_cluster_dataset(rows, cluster_tok, seq_points=SEQ_POINTS, osm_extractor=None, label=''):
    ids,aux,meta,osm_feats,y=[],[],[],[],[]
    for idx,r in enumerate(rows):
        pts=r['poly'][:seq_points]
        if len(pts)<seq_points: continue
        if idx%2000==0 and label: print(f"    {label}: {idx}/{len(rows)}",flush=True)
        ci,ca,co=[],[],[]
        for i,(lon,lat) in enumerate(pts):
            cid,rx,ry=cluster_tok.encode(lon,lat); ci.append(cid+1); ca.append([rx,ry])
            if osm_extractor is not None:
                pfx=f'p{i}'
                of=osm_extractor.point_features(lon,lat,pfx)
                co.append([of[f'{pfx}_nearest_major_m']/1500.0, of[f'{pfx}_nearest_secondary_m']/1500.0,
                           of[f'{pfx}_local_road_count_r200']/20.0, of[f'{pfx}_major_road_count_r300']/10.0,
                           of[f'{pfx}_node_density_r200']/50.0])
        ids.append(ci); aux.append(ca); meta.append(build_meta(pts,r['ts'],r['call'])); y.append(r['dur'])
        if osm_extractor is not None: osm_feats.append(co)
    ids=np.array(ids,dtype=np.int64); aux=np.array(aux,dtype=np.float32)
    meta=np.array(meta,dtype=np.float32); y=np.array(y,dtype=np.float32)
    osm_arr = np.array(osm_feats,dtype=np.float32) if osm_extractor is not None else None
    return ids, aux, meta, osm_arr, y

# ── Models ───────────────────────────────────────────────────────────
class ContSeqTransformer(nn.Module):
    """Continuous token SeqTransformer, optionally with OSM → token embedding."""
    def __init__(self, tok_dim, meta_dim, osm_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.tp=nn.Linear(tok_dim,d); self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
        self.use_osm = osm_dim > 0
        if self.use_osm: self.op = nn.Linear(osm_dim, d)
    def forward(self, tok, meta, osm=None):
        x=self.tp(tok)
        if self.use_osm and osm is not None: x=x+self.op(osm)
        cls=self.cls.expand(tok.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,pool,mf],1)).squeeze(1)

class TokenSeqTransformer(nn.Module):
    """Grid/Cluster token SeqTransformer, optionally with OSM → token embedding."""
    def __init__(self, vocab, aux_dim, meta_dim, osm_dim, max_len, d=64, nh=4, nl=2):
        super().__init__()
        self.emb=nn.Embedding(vocab+1,d,padding_idx=0); self.ap=nn.Linear(aux_dim,d)
        self.mp=nn.Sequential(nn.Linear(meta_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pe=nn.Parameter(torch.zeros(1,max_len+1,d))
        el=nn.TransformerEncoderLayer(d,nh,dim_feedforward=d*3,dropout=0.12,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.norm=nn.LayerNorm(d)
        self.ps=nn.Linear(d,1); self.head=nn.Sequential(nn.Linear(d*3,96),nn.GELU(),nn.Dropout(0.12),nn.Linear(96,1))
        self.use_osm = osm_dim > 0
        if self.use_osm: self.op = nn.Linear(osm_dim, d)
    def forward(self, ids, aux, meta, osm=None):
        x=self.emb(ids)+self.ap(aux)
        if self.use_osm and osm is not None: x=x+self.op(osm)
        cls=self.cls.expand(ids.size(0),-1,-1)
        x=torch.cat([cls,x],1); x=x+self.pe[:,:x.size(1),:]
        x=self.enc(x); cf=self.norm(x[:,0,:]); tf=self.norm(x[:,1:,:])
        att=torch.softmax(self.ps(tf),1); pool=torch.sum(tf*att,1)
        mf=self.mp(meta); return self.head(torch.cat([cf,pool,mf],1)).squeeze(1)

# ── Training ─────────────────────────────────────────────────────────
def train_cont(Xt, Xm, Xo, y, label, osm_dim, ep=EPOCHS, bs=BATCH_SIZE, patience=PATIENCE):
    n=len(Xt); idx=np.arange(n)
    tr_i,te_i=train_test_split(idx,test_size=0.2,random_state=42)
    tr_i,va_i=train_test_split(tr_i,test_size=0.1,random_state=42)
    y_log=np.log1p(y); ytr, yva, yte=y_log[tr_i],y_log[va_i],y_log[te_i]
    tok_mean=Xt[tr_i].mean(axis=(0,1)); tok_std=Xt[tr_i].std(axis=(0,1))+1e-6
    meta_mean=Xm[tr_i].mean(0); meta_std=Xm[tr_i].std(0)+1e-6
    Xt=(Xt-tok_mean)/tok_std; Xm=(Xm-meta_mean)/meta_std
    if Xo is not None:
        osm_mean=Xo[tr_i].mean(axis=(0,1)); osm_std=Xo[tr_i].std(axis=(0,1))+1e-6
        Xo=(Xo-osm_mean)/osm_std
    Xtr_t=torch.tensor(Xt[tr_i]); Xva_t=torch.tensor(Xt[va_i]); Xte_t=torch.tensor(Xt[te_i])
    Xtr_m=torch.tensor(Xm[tr_i]); Xva_m=torch.tensor(Xm[va_i]); Xte_m=torch.tensor(Xm[te_i])
    ytr_t=torch.tensor(ytr); yva_t=torch.tensor(yva); yte_t=torch.tensor(yte)
    Xtr_o=torch.tensor(Xo[tr_i]) if Xo is not None else None
    Xva_o=torch.tensor(Xo[va_i]) if Xo is not None else None
    Xte_o=torch.tensor(Xo[te_i]) if Xo is not None else None
    model=ContSeqTransformer(Xt.shape[2],Xm.shape[1],osm_dim,Xt.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); best_va=float('inf'); best_st=None; wait=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_t))
        for i in range(0,len(Xtr_t),bs):
            b=perm[i:i+bs]; o=Xtr_o[b] if Xtr_o is not None else None
            loss=crit(model(Xtr_t[b],Xtr_m[b],o),ytr_t[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            o=Xva_o if Xva_o is not None else None
            va_loss=float(crit(model(Xva_t,Xva_m,o),yva_t).item())
        sched.step(va_loss)
        if va_loss<best_va-1e-4: best_va=va_loss; best_st={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pred=np.expm1(model(Xte_t,Xte_m,Xte_o).cpu().numpy())
    true=np.expm1(yte_t.cpu().numpy())
    mae=np.mean(np.abs(pred-true)); rmse=np.sqrt(np.mean((pred-true)**2))
    mape=np.mean(np.abs((pred-true)/np.maximum(true,1)))*100
    print(f"  {label}: MAE={mae:.1f}s RMSE={rmse:.1f}s MAPE={mape:.1f}%")
    return {'mae':mae,'rmse':rmse,'mape':mape}

def train_token(ids_t,aux_t,meta_t,osm_t,y_t,label,vocab,osm_dim,ep=EPOCHS,bs=BATCH_SIZE,patience=PATIENCE):
    n=len(ids_t); idx=np.arange(n)
    tr_i,te_i=train_test_split(idx,test_size=0.2,random_state=42)
    tr_i,va_i=train_test_split(tr_i,test_size=0.1,random_state=42)
    y_log=np.log1p(y_t); ytr, yva, yte=y_log[tr_i],y_log[va_i],y_log[te_i]
    aux_t=(aux_t-aux_t[tr_i].mean(axis=(0,1)))/(aux_t[tr_i].std(axis=(0,1))+1e-6)
    meta_t=(meta_t-meta_t[tr_i].mean(0))/(meta_t[tr_i].std(0)+1e-6)
    if osm_t is not None:
        osm_t=(osm_t-osm_t[tr_i].mean(axis=(0,1)))/(osm_t[tr_i].std(axis=(0,1))+1e-6)
    Xtr_i=torch.tensor(ids_t[tr_i],dtype=torch.long); Xva_i=torch.tensor(ids_t[va_i],dtype=torch.long); Xte_i=torch.tensor(ids_t[te_i],dtype=torch.long)
    Xtr_a=torch.tensor(aux_t[tr_i]); Xva_a=torch.tensor(aux_t[va_i]); Xte_a=torch.tensor(aux_t[te_i])
    Xtr_m=torch.tensor(meta_t[tr_i]); Xva_m=torch.tensor(meta_t[va_i]); Xte_m=torch.tensor(meta_t[te_i])
    ytr_t=torch.tensor(ytr); yva_t=torch.tensor(yva); yte_t=torch.tensor(yte)
    Xtr_o=torch.tensor(osm_t[tr_i]) if osm_t is not None else None
    Xva_o=torch.tensor(osm_t[va_i]) if osm_t is not None else None
    Xte_o=torch.tensor(osm_t[te_i]) if osm_t is not None else None
    model=TokenSeqTransformer(vocab,aux_t.shape[2],meta_t.shape[1],osm_dim,ids_t.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.6,patience=1)
    crit=nn.SmoothL1Loss(beta=0.2); best_va=float('inf'); best_st=None; wait=0
    for e in range(1,ep+1):
        model.train(); perm=torch.randperm(len(Xtr_i))
        for i in range(0,len(Xtr_i),bs):
            b=perm[i:i+bs]; o=Xtr_o[b] if Xtr_o is not None else None
            loss=crit(model(Xtr_i[b],Xtr_a[b],Xtr_m[b],o),ytr_t[b])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            o=Xva_o if Xva_o is not None else None
            va_loss=float(crit(model(Xva_i,Xva_a,Xva_m,o),yva_t).item())
        sched.step(va_loss)
        if va_loss<best_va-1e-4: best_va=va_loss; best_st={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if wait>=patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pred=np.expm1(model(Xte_i,Xte_a,Xte_m,Xte_o).cpu().numpy())
    true=np.expm1(yte_t.cpu().numpy())
    mae=np.mean(np.abs(pred-true)); rmse=np.sqrt(np.mean((pred-true)**2))
    mape=np.mean(np.abs((pred-true)/np.maximum(true,1)))*100
    print(f"  {label}: MAE={mae:.1f}s RMSE={rmse:.1f}s MAPE={mape:.1f}%")
    return {'mae':mae,'rmse':rmse,'mape':mape}

# ── Main Experiment ─────────────────────────────────────────────────
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # 1. Load data
    rows = load_and_clean(N_SAMPLE)
    # 2. Initialize tokenizers
    grid_tok = GridTokenizer()
    cluster_tok = ClusterTokenizer(256)
    train_pts = [p for r in rows[:int(len(rows)*0.8)] for p in r['poly'][:SEQ_POINTS]]
    cluster_tok.fit(train_pts)
    print(f"  Grid vocab: {grid_tok.vocab}, Cluster: 256")
    # 3. Initialize OSM
    osm_path = 'data/porto_highways_bbox.osm.json'
    osm_ext = OSMFeatureExtractor(osm_path) if os.path.exists(osm_path) else None
    # 4. Build non-OSM datasets (fast, no OSM queries)
    print("\nBuilding non-OSM datasets...")
    t0 = time.time()
    cont_tok, cont_meta, _, cont_y = build_continuous_dataset(rows, label='cont')
    grid_ids, grid_aux, grid_meta, _, grid_y = build_grid_dataset(rows, grid_tok, label='grid')
    clust_ids, clust_aux, clust_meta, _, clust_y = build_cluster_dataset(rows, cluster_tok, label='clust')
    print(f"  Non-OSM datasets: {time.time()-t0:.0f}s")
    print(f"  Continuous: {len(cont_tok)}, Grid: {len(grid_ids)}, Cluster: {len(clust_ids)}")
    # 5. Build OSM datasets on subset (expensive per-point queries)
    print("\nBuilding OSM datasets (subset)...")
    osm_rows = rows[:min(N_OSM_SAMPLE, len(rows))]
    t0 = time.time()
    _, _, cont_osm, _ = build_continuous_dataset(osm_rows, osm_extractor=osm_ext, label='cont+osm')
    _, _, _, grid_osm, _ = build_grid_dataset(osm_rows, grid_tok, osm_extractor=osm_ext, label='grid+osm')
    _, _, _, clust_osm, _ = build_cluster_dataset(osm_rows, cluster_tok, osm_extractor=osm_ext, label='clust+osm')
    print(f"  OSM datasets: {time.time()-t0:.0f}s")
    # Subset non-OSM data to match OSM subset size (same filtering applied)
    n_osm = min(len(cont_osm), len(grid_osm), len(clust_osm))
    cont_tok_osm = cont_tok[:n_osm]
    cont_meta_osm = cont_meta[:n_osm]
    cont_y_osm = cont_y[:n_osm]
    grid_ids_osm = grid_ids[:n_osm]
    grid_aux_osm = grid_aux[:n_osm]
    grid_meta_osm = grid_meta[:n_osm]
    grid_y_osm = grid_y[:n_osm]
    clust_ids_osm = clust_ids[:n_osm]
    clust_aux_osm = clust_aux[:n_osm]
    clust_meta_osm = clust_meta[:n_osm]
    clust_y_osm = clust_y[:n_osm]
    # 6. Run experiments: 3 × 2
    results = {}
    print("\n" + "="*70)
    print("  RUNNING EXPERIMENTS: 3 tokenization × 2 OSM modes")
    print("="*70)
    print("\n── No OSM (full dataset) ──")
    results['cont_no_osm'] = train_cont(cont_tok, cont_meta, None, cont_y, "Continuous (no OSM)", 0)
    results['grid_no_osm'] = train_token(grid_ids, grid_aux, grid_meta, None, grid_y, "Grid Token (no OSM)", grid_tok.vocab, 0)
    results['clust_no_osm'] = train_token(clust_ids, clust_aux, clust_meta, None, clust_y, "Cluster Token (no OSM)", 256, 0)
    print("\n── No OSM (subset, for paired comparison) ──")
    results['cont_no_osm_sub'] = train_cont(cont_tok_osm, cont_meta_osm, None, cont_y_osm, "Continuous (no OSM, sub)", 0)
    results['grid_no_osm_sub'] = train_token(grid_ids_osm, grid_aux_osm, grid_meta_osm, None, grid_y_osm, "Grid Token (no OSM, sub)", grid_tok.vocab, 0)
    results['clust_no_osm_sub'] = train_token(clust_ids_osm, clust_aux_osm, clust_meta_osm, None, clust_y_osm, "Cluster Token (no OSM, sub)", 256, 0)
    print("\n── With OSM → token embedding ──")
    results['cont_osm'] = train_cont(cont_tok_osm, cont_meta_osm, cont_osm, cont_y_osm, "Continuous + OSM", OSM_DIM)
    results['grid_osm'] = train_token(grid_ids_osm, grid_aux_osm, grid_meta_osm, grid_osm, grid_y_osm, "Grid Token + OSM", grid_tok.vocab, OSM_DIM)
    results['clust_osm'] = train_token(clust_ids_osm, clust_aux_osm, clust_meta_osm, clust_osm, clust_y_osm, "Cluster Token + OSM", 256, OSM_DIM)

    # ── Ablation: different grid sizes and cluster counts (no OSM, full dataset) ──
    print("\n" + "="*70)
    print("  ABLATION: Grid size & Cluster count")
    print("="*70)
    # Grid 400m (coarse)
    print("\n── Grid 400m ──")
    gt_coarse = GridTokenizer(cl=0.0036, co=0.0044)
    gi_c, ga_c, gm_c, _, gy_c = build_grid_dataset(rows, gt_coarse, label='grid400m')
    results['grid400m'] = train_token(gi_c, ga_c, gm_c, None, gy_c, "Grid 400m", gt_coarse.vocab, 0)
    # Grid 100m (fine)
    print("\n── Grid 100m ──")
    gt_fine = GridTokenizer(cl=0.0009, co=0.0011)
    gi_f, ga_f, gm_f, _, gy_f = build_grid_dataset(rows, gt_fine, label='grid100m')
    results['grid100m'] = train_token(gi_f, ga_f, gm_f, None, gy_f, "Grid 100m", gt_fine.vocab, 0)
    # Cluster k=64 (coarse)
    print("\n── Cluster k=64 ──")
    ct64 = ClusterTokenizer(64)
    ct64.fit(train_pts)
    ci64, ca64, cm64, _, cy64 = build_cluster_dataset(rows, ct64, label='clust64')
    results['clust64'] = train_token(ci64, ca64, cm64, None, cy64, "Cluster k=64", 64, 0)
    # Cluster k=128
    print("\n── Cluster k=128 ──")
    ct128 = ClusterTokenizer(128)
    ct128.fit(train_pts)
    ci128, ca128, cm128, _, cy128 = build_cluster_dataset(rows, ct128, label='clust128')
    results['clust128'] = train_token(ci128, ca128, cm128, None, cy128, "Cluster k=128", 128, 0)
    # Cluster k=512 (fine)
    print("\n── Cluster k=512 ──")
    ct512 = ClusterTokenizer(512)
    ct512.fit(train_pts)
    ci512, ca512, cm512, _, cy512 = build_cluster_dataset(rows, ct512, label='clust512')
    results['clust512'] = train_token(ci512, ca512, cm512, None, cy512, "Cluster k=512", 512, 0)
    # 7. Summary
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY")
    print("="*70)
    print(f"  {'Method':<28s} {'N':>6s} {'OSM':>6s} {'MAE(s)':>8s} {'RMSE(s)':>9s} {'MAPE(%)':>8s}")
    print(f"  {'-'*62}")
    rows_summary = [
        ('cont_no_osm','Continuous',str(len(cont_tok)),'No'),
        ('grid_no_osm','Grid Token',str(len(grid_ids)),'No'),
        ('clust_no_osm','Cluster Token',str(len(clust_ids)),'No'),
        ('grid400m','Grid 400m   ',str(len(gi_c)),'No'),
        ('grid100m','Grid 100m   ',str(len(gi_f)),'No'),
        ('clust64','Cluster k=64',str(len(ci64)),'No'),
        ('clust128','Cluster k=128',str(len(ci128)),'No'),
        ('clust512','Cluster k=512',str(len(ci512)),'No'),
        ('cont_no_osm_sub','Continuous',str(len(cont_tok_osm)),'No (sub)'),
        ('cont_osm','Continuous',str(len(cont_tok_osm)),'Yes'),
        ('grid_no_osm_sub','Grid Token',str(len(grid_ids_osm)),'No (sub)'),
        ('grid_osm','Grid Token',str(len(grid_ids_osm)),'Yes'),
        ('clust_no_osm_sub','Cluster Token',str(len(clust_ids_osm)),'No (sub)'),
        ('clust_osm','Cluster Token',str(len(clust_ids_osm)),'Yes'),
    ]
    for k,name_clean,n_str,osm_s in rows_summary:
        r=results[k]
        print(f"  {name_clean:<28s} {n_str:>6s} {osm_s:>6s} {r['mae']:>8.1f} {r['rmse']:>9.1f} {r['mape']:>8.1f}")
    print("="*70)
    # Save results
    summary_path = 'output/task4_token_osm_summary.md'
    os.makedirs('output',exist_ok=True)
    with open(summary_path,'w') as f:
        f.write("# Task 4: Tokenization × OSM Cross Experiment\n\n")
        f.write(f"Config: {SEQ_POINTS}-point prefix, OSM subset = {N_OSM_SAMPLE}\n\n")
        f.write("## Full dataset (no OSM)\n\n")
        f.write(f"Samples: {len(cont_tok)}\n\n")
        f.write("| Method | MAE (s) | RMSE (s) | MAPE (%) |\n|---|---:|---:|---:|")
        for k,n in [('cont_no_osm','Continuous'),('grid_no_osm','Grid Token'),('clust_no_osm','Cluster Token')]:
            r=results[k]
            f.write(f"\n| {n} | {r['mae']:.1f} | {r['rmse']:.1f} | {r['mape']:.1f} |")
        f.write(f"\n\n## OSM ablation (subset = {N_OSM_SAMPLE})\n\n")
        f.write("| Method | OSM | MAE (s) | RMSE (s) | MAPE (%) | Δ MAE |\n|---|---:|---:|---:|---:|---:|\n")
        for k,name_clean in [('cont_no_osm_sub','Continuous'),('cont_osm','Continuous'),
                             ('grid_no_osm_sub','Grid Token'),('grid_osm','Grid Token'),
                             ('clust_no_osm_sub','Cluster Token'),('clust_osm','Cluster Token')]:
            r=results[k]
            osm_s='No' if '_no_osm_sub' in k else 'Yes'
            delta = ''
            f.write(f"| {name_clean} | {osm_s} | {r['mae']:.1f} | {r['rmse']:.1f} | {r['mape']:.1f} | {delta} |\n")
        # Ablation section
    f.write("\n## Ablation: Grid size & Cluster count\n\n")
    f.write("| Method | Vocab | MAE (s) | RMSE (s) | MAPE (%) |\n|---|---:|---:|---:|---:|\n")
    for k,n,v in [('grid400m','Grid 400m',gt_coarse.vocab),('grid_no_osm','Grid 200m',grid_tok.vocab),
                  ('grid100m','Grid 100m',gt_fine.vocab),('clust64','Cluster k=64',64),
                  ('clust128','Cluster k=128',128),('clust_no_osm','Cluster k=256',256),('clust512','Cluster k=512',512)]:
        r=results[k]
        f.write(f"| {n} | {v} | {r['mae']:.1f} | {r['rmse']:.1f} | {r['mape']:.1f} |\n")
    print(f"\nResults saved to {summary_path}")

if __name__=="__main__":
    main()
