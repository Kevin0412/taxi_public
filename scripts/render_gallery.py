import os,subprocess,shutil
from pathlib import Path
from PIL import Image
os.environ["MPLBACKEND"]="Agg"
os.environ["OMP_NUM_THREADS"]="2"
out=Path("render-output");out.mkdir(exist_ok=True)
preview=Path("docs/generated");preview.mkdir(parents=True,exist_ok=True)
subprocess.run(["python","taxi_duration_prediction.py","--mode","synthetic","--n","800","--skip-prefix-analysis","--skip-sequence-experiment"],check=True)
for p in Path("output").glob("*"):
    if p.is_file():shutil.copy2(p,out/p.name)
im=Image.open(out/"taxi_02_model_comparison.png");im.thumbnail((1100,900));im.save(preview/"preview.webp",quality=88)
