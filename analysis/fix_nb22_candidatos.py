"""Choose the candidate frames before copying them, not after.

CANDIDATOS_POR_OBJETIVO only narrowed the ranking, so the whole lateral pool was
copied and scored whatever its value, and the switch bought nothing where it was
needed. Sampling the candidates from the manifest first makes the archive, the copy
and the inference all shrink with it. Copying is what costs here: the archive moves
about two files a second off Drive, so the full pool is close to ten hours and a
sixfold candidate set is three.

The copy also skips what is already on local disk, so an interrupted run continues
instead of starting over.
"""

import io
import json
import os
import shutil
import sys

RUTA = (r"G:\My Drive\UNM_vertebras_seg_v3\tesis_seg\notebooks"
        r"\22_unm_pool_incertidumbre.ipynb")
COPIA = RUTA + ".ANTES_candidatos"

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    crudo = fh.read()
if crudo.count("\r") != crudo.count("\r\n"):
    sys.exit("ABORTO: CR sueltos antes de tocar.")
nb = json.loads(crudo)

P2 = '''# === PASO 2: puntuar y construir los pools ===
import shutil, subprocess
import pandas as pd

os.makedirs(os.path.dirname(CSV_ENTROPIA), exist_ok=True)

# ---------- 2a. elegir los candidatos ANTES de copiar ----------
# Copiar es lo que cuesta: el archivo sale de Drive a unos 2 frames/s, asi que el pool
# entero son ~10 h y un candidato de 6x son ~3 h. Reducir aqui reduce copia, inferencia
# y tiempo; reducir despues no reducia nada.
_rng_cand = np.random.RandomState(7)
if CANDIDATOS_POR_OBJETIVO:
    _por_v = collections.defaultdict(list)
    for _f in TODOS:
        _por_v[_f.split("_")[0]].append(_f)
    CANDIDATOS = []
    for _v, _k in K_OBJETIVO.items():
        _disp = _por_v[_v]
        _n = min(len(_disp), _k * CANDIDATOS_POR_OBJETIVO)
        _idx = _rng_cand.choice(len(_disp), _n, replace=False)
        CANDIDATOS += [_disp[i] for i in sorted(_idx)]
    CANDIDATOS.sort()
else:
    CANDIDATOS = list(TODOS)

print("candidatos a copiar y puntuar: %d de %d  (%.1f x el objetivo)"
      % (len(CANDIDATOS), len(TODOS), len(CANDIDATOS) / sum(K_OBJETIVO.values())))
_cv = collections.Counter(f.split("_")[0] for f in CANDIDATOS)
_pocos = [v for v in K_OBJETIVO if _cv.get(v, 0) < K_OBJETIVO[v]]
assert not _pocos, f"FALLO: {_pocos} se quedan sin candidatos suficientes"
print("   todos los videos tienen candidatos de sobra   OK")
print()

# ---------- 2b. entropia por frame ----------
_ya = (os.path.isfile(CSV_ENTROPIA)
       and len(pd.read_csv(CSV_ENTROPIA)) >= len(CANDIDATOS) - 10)
if _ya:
    print("el csv de entropia ya esta, con", len(pd.read_csv(CSV_ENTROPIA)),
          "filas. Se salta el calculo.")
else:
    os.makedirs(LOCAL, exist_ok=True)
    _tengo = set(os.listdir(LOCAL))
    _faltan = [f for f in CANDIDATOS if f not in _tengo]
    print("en local ya hay %d; faltan %d" % (len(_tengo), len(_faltan)))
    if _faltan:
        # 32 hilos en vez de tar: la latencia de Drive es de red, no de ancho de
        # banda, asi que leer en paralelo la esconde. Medido en esta misma sesion:
        # tar en serie 2.1 archivos/s, 32 hilos 45.9 archivos/s.
        from concurrent.futures import ThreadPoolExecutor
        HILOS = 32

        def _copiar(f):
            try:
                shutil.copy2(os.path.join(POOL_TODOS, f), os.path.join(LOCAL, f))
                return 1
            except Exception:
                return 0

        print("copiando %d archivos con %d hilos..." % (len(_faltan), HILOS))
        _t0, _ok = time.time(), 0
        with ThreadPoolExecutor(HILOS) as _ex:
            for _n, _r in enumerate(_ex.map(_copiar, _faltan), 1):
                _ok += _r
                if _n % 5000 == 0:
                    _v = _n / (time.time() - _t0)
                    print("   %6d / %d   %.1f arch/s   faltan %.1f min"
                          % (_n, len(_faltan), _v, (len(_faltan) - _n) / _v / 60))
        _dt = (time.time() - _t0) / 60
        print("   copiados %d de %d en %.1f min (%.1f arch/s)"
              % (_ok, len(_faltan), _dt, _ok / max(1e-9, _dt * 60)))
    _tengo = set(os.listdir(LOCAL))
    _listos = [f for f in CANDIDATOS if f in _tengo]
    print("candidatos en local:", len(_listos), "de", len(CANDIDATOS))
    assert len(_listos) >= len(CANDIDATOS) * 0.99, "FALLO: falto demasiado por copiar"

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
            if _i % 4000 < 32:
                print("   %6d / %d   %.1f min" % (_i, len(_dsE), (time.time() - _t0) / 60))
    pd.DataFrame(_filas, columns=["stem", "H_mean"]).to_csv(CSV_ENTROPIA, index=False)
    print("guardado:", CSV_ENTROPIA, len(_filas), "filas")

ENT = pd.read_csv(CSV_ENTROPIA)
ENT["video"] = ENT["stem"].str.extract(r"^(v\\d+)_")
print("entropia: %d frames, %d videos, H entre %.4f y %.4f"
      % (len(ENT), ENT.video.nunique(), ENT.H_mean.min(), ENT.H_mean.max()))

# ---------- 2c. armar los dos pools, POR VIDEO ----------
# ENT ya contiene SOLO los candidatos, asi que aqui se ordena y se corta, sin volver
# a muestrear.
for _nombre, _mayor in [(POOL_INCIERTOS, True), (POOL_CIERTOS, False)]:
    _dst = f"{BASE}/{_nombre}/images"
    if os.path.isdir(_dst) and len([f for f in os.listdir(_dst) if f.endswith(".png")]) \\
            == sum(K_OBJETIVO.values()):
        print("ya esta:", _nombre)
        continue
    os.makedirs(_dst, exist_ok=True)
    _elegidos = []
    for _v, _k in K_OBJETIVO.items():
        _sub = ENT[ENT.video == _v].sort_values("H_mean", ascending=not _mayor)
        assert len(_sub) >= _k, f"FALLO: {_v} tiene {len(_sub)} candidatos y hacen falta {_k}"
        _elegidos += list(_sub.head(_k).stem)
    for _s in _elegidos:
        shutil.copy2(os.path.join(LOCAL, _s), os.path.join(_dst, _s))
    print("%s -> %d frames" % (_nombre, len(_elegidos)))

# ---------- 2d. las puertas del diseno ----------
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
_hi = ENT.set_index("stem").loc[list(_a), "H_mean"].mean()
_lo = ENT.set_index("stem").loc[list(_b), "H_mean"].mean()
print("  H media del pool incierto : %.4f" % _hi)
print("  H media del pool cierto   : %.4f" % _lo)
print("  separacion                : %.4f" % (_hi - _lo))
print("  OK: mismo tamano, misma composicion por video, sin solapamiento.")
'''

nueva = P2.split("\n")
nb["cells"][9]["source"] = ([l + "\n" for l in nueva[:-1]]
                            + ([nueva[-1]] if nueva[-1] else []))

if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
salida = json.dumps(nb, indent=1, ensure_ascii=False)
salida = salida.replace("\r\n", "\n").replace("\n", "\r\n")
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(salida)
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    escrito = fh.read()
rel = json.loads(escrito)

print("celdas       :", len(rel["cells"]))
print("CR sueltos   :", escrito.count("\r") - escrito.count("\r\n"))
print("LF sueltos   :", escrito.count("\n") - escrito.count("\r\n"))
print("reduce antes de copiar:", "CANDIDATOS = []" in escrito)
print("copia solo lo que falta:", "solo lo que falta" in escrito)

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
