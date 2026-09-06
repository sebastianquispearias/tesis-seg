"""Stop notebook 22 from ever enumerating the lateral pool on Drive.

The Drive FUSE mount in Colab raises Errno 5 when asked to list a directory with tens
of thousands of entries, which is what stopped the second gate. Every readdir on that
folder is replaced by a read of a manifest written from a machine that can list it,
including the one hidden inside UnlabeledFramesDataset, which calls list_png_files on
the directory it is given. The archive is then built from the manifest with tar -T
rather than by walking the directory.
"""

import io
import json
import os
import shutil
import sys

RUTA = (r"G:\My Drive\UNM_vertebras_seg_v3\tesis_seg\notebooks"
        r"\22_unm_pool_incertidumbre.ipynb")
COPIA = RUTA + ".ANTES_manifiesto"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    crudo = fh.read()
if crudo.count("\r") != crudo.count("\r\n"):
    sys.exit("ABORTO: CR sueltos antes de tocar.")
nb = json.loads(crudo)
celdas_antes = len(nb["cells"])

P1 = '''# ============================================================
# VERIFICACION - NO ENTRENA, NO PUNTUA EL POOL. ~3 minutos.
# ============================================================
import collections, glob, os, re, time
import cv2
import numpy as np
import torch

BASE    = "/content/drive/MyDrive/UNM_vertebras_seg_v3"
ROTULOS = "/content/drive/MyDrive/UNM TalkBank Dysphagia/rotulos"

POOL_TEMPORAL  = f"{BASE}/unlabeling_r10_max0/images"
POOL_ALEATORIO = f"{BASE}/unlabeling_std_matched_r10/images"
POOL_TODOS     = f"{BASE}/unlabeling_all_lateral/images"

# El pool lateral tiene 74.774 archivos y el montaje de Drive en Colab NO PUEDE
# LISTARLO: os.listdir, glob y tar sobre la carpeta fallan con [Errno 5]. Los nombres
# se leen de un manifiesto escrito desde una maquina que si puede listarla.
MANIFIESTO = f"{BASE}/unlabeling_all_lateral/manifest_all_lateral.txt"

# el juez: un U-Net++ supervisado entrenado en esta misma tanda, con el ruido arreglado
JUEZ = f"{BASE}/runs_nnunet_ablation/A_baseline_fixnoise/seed_0/best_model.pt"

print("PUERTA 1: el ruido de la augmentation es el pedido")
_gn = [t for t in get_supervised_train_augmentation({}).transforms
       if t.__class__.__name__ == "GaussNoise"][0]
_sr = _gn.to_dict()["transform"]["std_range"]
assert _sr[1] < 0.05, ("CODIGO VIEJO: std_range es el default de la 2.x. "
                       "Borra /content/tesis-seg, vuelve a clonar y reinicia.")
print("   std %.2f a %.2f niveles sobre 255   OK" % (_sr[0] * 255, _sr[1] * 255))
print()

print("PUERTA 2: los pools estan y tienen el tamano esperado")
assert os.path.isfile(MANIFIESTO), (
    f"FALTA el manifiesto {MANIFIESTO}. Generarlo con "
    "tesis_seg/analysis/manifiesto_lateral.py desde una maquina que pueda listar "
    "la carpeta.")
with open(MANIFIESTO) as _fh:
    TODOS = [l.strip() for l in _fh if l.strip().endswith(".png")]
print("   todos laterales  %6d frames  (del manifiesto, sin listar la carpeta)"
      % len(TODOS))
for _nom, _d in [("temporal r10", POOL_TEMPORAL), ("aleatorio r10", POOL_ALEATORIO)]:
    assert os.path.isdir(_d), f"FALTA {_d}"
    _k = len([f for f in os.listdir(_d) if f.endswith(".png")])
    print("   %-16s %6d frames" % (_nom, _k))
assert os.path.isdir(POOL_TODOS), f"FALTA {POOL_TODOS}"
assert len(TODOS) == 74774, f"el manifiesto trae {len(TODOS)} nombres, no 74774"
print("   OK")
print()

print("PUERTA 3: el juez existe y CARGA con el torch de hoy")
assert os.path.isfile(JUEZ), f"FALTA el checkpoint juez: {JUEZ}"
from src.models import create_model
from src.preprocessing import preprocess_image_and_mask
_m = create_model("unetpp", "efficientnet-b3", 1)
_sd = torch.load(JUEZ, map_location="cpu", weights_only=False)
if isinstance(_sd, dict) and "model_state_dict" in _sd:
    _sd = _sd["model_state_dict"]
elif isinstance(_sd, dict) and "state_dict" in _sd:
    _sd = _sd["state_dict"]
_faltan = _m.load_state_dict(_sd, strict=False)
print("   claves que faltan :", len(_faltan.missing_keys))
print("   claves sobrantes  :", len(_faltan.unexpected_keys))
assert len(_faltan.missing_keys) == 0, "FALLO: el checkpoint no encaja con el modelo"
print("   OK: carga limpio")
print()

print("PUERTA 4: los conteos POR VIDEO que hay que igualar")
def _por_video(nombres):
    c = collections.Counter()
    for f in nombres:
        m = re.match(r"(v\\d+)_f\\d+\\.png$", f)
        if m:
            c[m.group(1)] += 1
    return c

K_OBJETIVO = _por_video(os.listdir(POOL_TEMPORAL))
_alea = _por_video(os.listdir(POOL_ALEATORIO))
_dif = [v for v in K_OBJETIVO if K_OBJETIVO[v] != _alea.get(v)]
print("   videos                    :", len(K_OBJETIVO))
print("   suma de k_v               :", sum(K_OBJETIVO.values()))
print("   videos donde temporal y aleatorio NO coinciden:", len(_dif))
assert not _dif, "FALLO: el control aleatorio no iguala por video; revisar el diseno"
print("   OK: el aleatorio iguala por video, asi que el de incertidumbre tambien debe")
print()

print("PUERTA 5: hay candidatos de sobra en cada video")
_cand = _por_video(TODOS)
_pocos = [v for v in K_OBJETIVO if _cand.get(v, 0) < K_OBJETIVO[v]]
print("   videos en el pool lateral :", len(_cand))
print("   videos con menos candidatos que su objetivo:", len(_pocos))
assert not _pocos, f"FALLO: {_pocos} no tienen suficientes candidatos"
print("   OK")
print()

print("PUERTA 6: CUANTO TARDA DE VERDAD. Esto es lo que decide el Paso 2.")
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_m = _m.to(_dev).eval()
_muestra = TODOS[::len(TODOS) // 120][:120]
_t0 = time.time()
_leidos = 0
with torch.no_grad():
    for _f in _muestra:
        _im = cv2.imread(os.path.join(POOL_TODOS, _f), cv2.IMREAD_COLOR)
        if _im is None:
            continue
        _im = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB)
        _pr = preprocess_image_and_mask(
            image_uint8=_im, mask_uint8=np.zeros(_im.shape[:2], np.uint8),
            target_size=(320, 320), use_pad=True, imagenet_norm=False,
            image_preproc="base", image_norm="unit", mask_smoothing="none")
        _x = torch.from_numpy(_pr["image"]).float().unsqueeze(0).to(_dev)
        _ = torch.sigmoid(_m(_x))
        _leidos += 1
_dt = time.time() - _t0
_vel = _leidos / _dt
print("   %d frames en %.1f s  ->  %.1f frames/s LEYENDO DE DRIVE" % (_leidos, _dt, _vel))
print("   los %d del pool tardarian: %.0f min (%.1f h)"
      % (len(TODOS), len(TODOS) / _vel / 60, len(TODOS) / _vel / 3600))
print()
if len(TODOS) / _vel / 3600 > 2.0:
    print("   >>> Desde Drive no sale a cuenta. El Paso 2 hara tar -T + local.")
    print("       Si aun asi sale caro, baja CANDIDATOS_POR_OBJETIVO en el Paso 1b.")
else:
    print("   >>> Se podria leer directo de Drive, pero el Paso 2 usa local igual.")
print()
print("TODO VERIFICADO.")
'''

P2 = '''# === PASO 2: puntuar y construir los pools ===
import shutil, subprocess
import pandas as pd

os.makedirs(os.path.dirname(CSV_ENTROPIA), exist_ok=True)

# ---------- 2a. entropia por frame ----------
_ya = (os.path.isfile(CSV_ENTROPIA)
       and len(pd.read_csv(CSV_ENTROPIA)) >= len(TODOS) - 10)
if _ya:
    print("entropy_all_lateral.csv ya esta, con",
          len(pd.read_csv(CSV_ENTROPIA)), "filas. Se salta el calculo.")
else:
    _n_local = len(os.listdir(LOCAL)) if os.path.isdir(LOCAL) else 0
    if _n_local < len(TODOS):
        print("copiando el pool al SSD local con tar -T (sin listar la carpeta)...")
        os.makedirs(LOCAL, exist_ok=True)
        _lista = "/content/manifest_lateral.txt"
        shutil.copy2(MANIFIESTO, _lista)
        _t0 = time.time()
        _cmd = (f'tar -cf - -C "{POOL_TODOS}" -T "{_lista}" '
                f'| tar -xf - -C "{LOCAL}"')
        _r = subprocess.run(_cmd, shell=True)
        print("   codigo de salida: %d,  %.1f min"
              % (_r.returncode, (time.time() - _t0) / 60))
        _n_local = len(os.listdir(LOCAL))
    print("frames en local:", _n_local)
    assert _n_local >= len(TODOS) * 0.99, (
        f"FALLO: solo {_n_local} de {len(TODOS)} llegaron al disco local")

    _fs = sorted(f for f in os.listdir(LOCAL) if f.endswith(".png"))
    _cfgE = get_default_config()
    _cfgE.update({"target_size": (320, 320), "use_pad": True, "imagenet_norm": False,
                  "image_preproc": "base", "num_workers": 4})
    from src.datasets import UnlabeledFramesDataset
    _dsE = UnlabeledFramesDataset(images_dir=LOCAL, cfg=_cfgE)
    _ldE = torch.utils.data.DataLoader(_dsE, batch_size=32, shuffle=False,
                                       num_workers=4, pin_memory=True)
    _m = _m.to(_dev).eval()
    _filas, _i = [], 0
    _t0 = time.time()
    with torch.no_grad():
        for _b in _ldE:
            _x = _b["weak_image"].float().to(_dev)
            _p = torch.sigmoid(_m(_x)).clamp(1e-6, 1 - 1e-6)
            _h = -(_p * _p.log() + (1 - _p) * (1 - _p).log())      # entropia binaria
            _hm = _h.mean(dim=(1, 2, 3)).cpu().numpy()
            for _k, _nm in enumerate(_b["name"]):
                _filas.append((_nm, float(_hm[_k])))
                _i += 1
            if _i % 8000 < 32:
                print("   %6d / %d   %.1f min" % (_i, len(_dsE), (time.time() - _t0) / 60))
    pd.DataFrame(_filas, columns=["stem", "H_mean"]).to_csv(CSV_ENTROPIA, index=False)
    print("guardado:", CSV_ENTROPIA, len(_filas), "filas")

ENT = pd.read_csv(CSV_ENTROPIA)
ENT["video"] = ENT["stem"].str.extract(r"^(v\\d+)_")
print("entropia: %d frames, %d videos, H entre %.4f y %.4f"
      % (len(ENT), ENT.video.nunique(), ENT.H_mean.min(), ENT.H_mean.max()))

# ---------- 2b. armar los dos pools, POR VIDEO ----------
_rng = np.random.RandomState(42)
_origen = LOCAL if os.path.isdir(LOCAL) else POOL_TODOS
for _nombre, _mayor in [(POOL_INCIERTOS, True), (POOL_CIERTOS, False)]:
    _dst = f"{BASE}/{_nombre}/images"
    if os.path.isdir(_dst) and len([f for f in os.listdir(_dst) if f.endswith(".png")]) \\
            == sum(K_OBJETIVO.values()):
        print("ya esta:", _nombre)
        continue
    os.makedirs(_dst, exist_ok=True)
    _elegidos = []
    for _v, _k in K_OBJETIVO.items():
        _sub = ENT[ENT.video == _v]
        if CANDIDATOS_POR_OBJETIVO:
            _n = min(len(_sub), _k * CANDIDATOS_POR_OBJETIVO)
            _sub = _sub.iloc[_rng.choice(len(_sub), _n, replace=False)]
        _sub = _sub.sort_values("H_mean", ascending=not _mayor)
        assert len(_sub) >= _k, f"FALLO: {_v} tiene {len(_sub)} candidatos y hacen falta {_k}"
        _elegidos += list(_sub.head(_k).stem)
    for _s in _elegidos:
        shutil.copy2(os.path.join(_origen, _s), os.path.join(_dst, _s))
    print("%s -> %d frames" % (_nombre, len(_elegidos)))

# ---------- 2c. las puertas del diseno ----------
print()
print("PUERTAS DEL DISENO")
for _nombre in [POOL_INCIERTOS, POOL_CIERTOS]:
    _d = f"{BASE}/{_nombre}/images"
    _c = _por_video(os.listdir(_d))
    _mal = [v for v in K_OBJETIVO if K_OBJETIVO[v] != _c.get(v, 0)]
    _tot = sum(_c.values())
    print("  %-36s %5d frames | videos mal igualados: %d" % (_nombre, _tot, len(_mal)))
    assert _tot == sum(K_OBJETIVO.values()), "FALLO: tamano distinto al objetivo"
    assert not _mal, "FALLO: no iguala por video"
_a = set(os.listdir(f"{BASE}/{POOL_INCIERTOS}/images"))
_b = set(os.listdir(f"{BASE}/{POOL_CIERTOS}/images"))
print("  solapamiento entre los dos pools:", len(_a & _b), "(tiene que ser 0)")
assert not (_a & _b), "FALLO: los dos pools comparten frames"
print("  OK: mismo tamano, misma composicion por video, sin solapamiento.")
'''


def fuente(txt):
    L = txt.split("\n")
    return [l + "\n" for l in L[:-1]] + ([L[-1]] if L[-1] else [])


nb["cells"][3]["source"] = fuente(P1)
nb["cells"][9]["source"] = fuente(P2)

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
salida = json.dumps(nb, indent=1, ensure_ascii=False)
salida = salida.replace("\r\n", "\n").replace("\n", "\r\n")
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(salida)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()
rel = json.loads(escrito)

print("celdas            :", celdas_antes, "->", len(rel["cells"]))
print("CR sueltos        :", escrito.count("\r") - escrito.count("\r\n"))
print("LF sueltos        :", escrito.count("\n") - escrito.count("\r\n"))
todo = escrito
print("queda os.listdir(POOL_TODOS):", "listdir(POOL_TODOS)" in todo)
print("queda glob sobre POOL_TODOS :", "glob" in todo and "POOL_TODOS)" in todo
      and "glob.glob(POOL_TODOS" in todo)
print("usa el manifiesto           :", todo.count("MANIFIESTO"), "veces")
print("tar con -T                  :", "-T" in todo)

malas = []
for i, c in enumerate(rel["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    limpio = "\n".join(("pass  # " + l if l[:1] in ("!", "%") else l)
                       for l in s.split("\n"))
    try:
        compile(limpio, "c%d" % i, "exec")
    except SyntaxError as e:
        malas.append((i, e.lineno, e.msg))
print("errores de sintaxis:", malas if malas else "ninguno")
