"""
anotar_categorias.py  —  INTERACTIVE annotation + render for the category figure.

Two modes:
  python anotar_categorias.py annotate   # click arrows/circles on each panel -> saves JSON
  python anotar_categorias.py render      # reads JSON -> figura_categorias.pdf/.png

Layout of the final figure: 5 rows (A-E), 2 columns [original frame | frame+mask].
Your arrows (solid red) and circles (dashed red) are drawn on the RIGHT column.
Privacy crop (fixed size, same zoom) drops equipment metadata + facial profile.

ANNOTATE controls (one panel at a time, right column shown):
  - press 'a'  -> ARROW mode: click TAIL, then click HEAD (solid red arrow)
  - press 'c'  -> CIRCLE mode: click CENTER, then click a point on the RIM (dashed red circle)
  - press 'u'  -> undo last annotation on this panel
  - press 'enter' or close the window -> go to next panel (auto-saved)
This script does NOT modify any original mask or frame.
"""
import sys, csv, json, os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT = r"G:/My Drive/UNM_vertebras_seg_v3"
LOCAL = r"C:/Users/User/temp_inter_rater/inca_fig"
METRICS = f"{PROJECT}/resultados/inter_rater_inca/figure_candidates/per_frame_metrics.csv"
OUT_DIR = f"{PROJECT}/resultados/inter_rater_inca/figuras_finales"
JSON_PATH = f"{OUT_DIR}/anotaciones_categorias.json"
CYAN = "#00E5FF"
RED = "#FF2222"
CH, CW = 560, 300          # fixed crop (same size & zoom in all panels)
CONN = np.ones((3, 3), int)
os.makedirs(OUT_DIR, exist_ok=True)

metrics = {r["video_frame"]: r for r in csv.DictReader(open(METRICS, encoding="utf-8"))}
bof = {r["video_frame"]: str(r["batch"])
       for r in csv.DictReader(open(f"{LOCAL}/video_frame_metadata.csv", encoding="utf-8"))}

def _bin(a):
    a = np.where(a > 127, 255, 0).astype(np.uint8)
    if (a == 255).sum() / a.size > 0.10:
        a = 255 - a
    return a > 127

def load_mask(vf, lab):
    b = bof[vf]
    return _bin(np.array(Image.open(f"{LOCAL}/rotulos/batch-{b}/{lab}/{vf}/Mask.tif").convert("L")))

def load_frame(vf, split):
    return np.array(Image.open(f"{LOCAL}/frames/{split}/{vf}.png").convert("L"))

def centroid(m):
    ys, xs = np.where(m); return ys.mean(), xs.mean()

def crop_box(cy, cx, H, W, oy=0, ox=0):
    y0 = int(cy - CH / 2) + oy; x0 = int(cx - CW / 2) + ox
    y0 = max(0, min(y0, H - CH)); x0 = max(0, min(x0, W - CW))
    return y0, y0 + CH, x0, x0 + CW

# category -> (frame, which mask to show). D uses ONE mask (D1). E shows NO mask.
# oy/ox = small crop offset for privacy (E shifted down to drop the timestamp corner).
PANELS = [
    ("A", "v39_f4", "less", 0, 0),
    ("B", "v207_f159", "more", 0, 0),
    ("C", "v235_f214", "A", 0, 0),
    ("D", "v128_f235", "invader", 0, 0),
    ("E", "v12_f90", "none", 60, -20),
]

def panel_image(cat, vf, sel, oy, ox):
    r = metrics[vf]; split = r["split"]; la, lb = r["labeler_A"], r["labeler_B"]
    fr = load_frame(vf, split); H, W = fr.shape
    mA = load_mask(vf, la); mB = load_mask(vf, lb)
    nA, nB = int(r["ncomp_A"]), int(r["ncomp_B"])
    if sel == "less":
        show = mA if nA < nB else mB
    elif sel == "more":
        show = mB if nB > nA else mA
    elif sel == "invader":
        xor = mA ^ mB; lbl, n = ndimage.label(xor, structure=CONN)
        if n:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            big = lbl == int(np.argmax(sizes)) + 1
            show = mA if (big & mA).sum() >= (big & mB).sum() else mB
        else:
            show = mA
    elif sel == "none":
        show = None
    else:
        show = mA
    ref = show if show is not None else (mA | mB)
    cy, cx = centroid(ref)
    y0, y1, x0, x1 = crop_box(cy, cx, H, W, oy, ox)
    crop = fr[y0:y1, x0:x1]
    scrop = show[y0:y1, x0:x1] if show is not None else None
    return crop, scrop

# ---------------- ANNOTATE ----------------
def annotate():
    matplotlib.use("TkAgg")  # interactive backend (falls back: try "QtAgg")
    # start from existing annotations if present, so you can refine one panel at a time
    data = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}
    for cat, vf, sel, oy, ox in PANELS:
        crop, scrop = panel_image(cat, vf, sel, oy, ox)
        prev = data.get(cat, {"arrows": [], "circles": []})
        state = {"arrows": [list(a) for a in prev.get("arrows", [])],
                 "circles": [list(c) for c in prev.get("circles", [])]}
        mode = {"m": "a"}; pending = {"pts": []}
        # TWO columns: left = reference frame (no mask); right = frame+mask (click here)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 8))

        def draw_right():
            axR.clear()
            axR.imshow(crop, cmap="gray")
            if scrop is not None:
                axR.contour(scrop, [0.5], colors=[CYAN], linewidths=1.2)
            for (x0, y0, x1, y1) in state["arrows"]:
                axR.annotate("", xy=(x1, y1), xytext=(x0, y0),
                             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
            for (cx, cy, rad) in state["circles"]:
                axR.add_patch(Circle((cx, cy), rad, fill=False, edgecolor=RED, lw=1.6, ls="--"))
            axR.set_title(f"CLICK HERE  mode={'ARROW' if mode['m']=='a' else 'CIRCLE'}", fontsize=9)
            axR.set_axis_off()
            fig.canvas.draw_idle()

        axL.imshow(crop, cmap="gray"); axL.set_title("reference (original frame)", fontsize=9)
        axL.set_axis_off()
        fig.suptitle(f"[{cat}] {vf}   a=arrow  c=circle  u=undo  Enter=next", fontsize=11)
        draw_right()

        def on_key(e):
            if e.key in ("a", "c"):
                mode["m"] = e.key; pending["pts"] = []; draw_right()
            elif e.key == "u":
                if pending["pts"]:
                    pending["pts"] = []
                elif mode["m"] == "a" and state["arrows"]:
                    state["arrows"].pop()
                elif mode["m"] == "c" and state["circles"]:
                    state["circles"].pop()
                draw_right()
            elif e.key == "enter":
                plt.close(fig)

        def on_click(e):
            if e.inaxes != axR or e.xdata is None:   # only clicks on the RIGHT panel count
                return
            pending["pts"].append((e.xdata, e.ydata))
            if len(pending["pts"]) == 2:
                (x0, y0), (x1, y1) = pending["pts"]; pending["pts"] = []
                if mode["m"] == "a":
                    state["arrows"].append([x0, y0, x1, y1])
                else:
                    state["circles"].append([x0, y0, float(np.hypot(x1 - x0, y1 - y0))])
                draw_right()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()
        data[cat] = state
        json.dump(data, open(JSON_PATH, "w"), indent=2, default=float)
        print(f"[{cat}] saved: {len(state['arrows'])} arrows, {len(state['circles'])} circles")
    print("Annotations saved to", JSON_PATH, "\nNow run:  python anotar_categorias.py render")

# ---------------- RENDER ----------------
def render():
    matplotlib.use("Agg")
    data = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(6, 3 * len(PANELS)))
    for (cat, vf, sel, oy, ox), axrow in zip(PANELS, axes):
        crop, scrop = panel_image(cat, vf, sel, oy, ox)
        axrow[0].imshow(crop, cmap="gray", interpolation="nearest"); axrow[0].set_axis_off()
        axrow[1].imshow(crop, cmap="gray", interpolation="nearest")
        if scrop is not None:
            axrow[1].contour(scrop, [0.5], colors=[CYAN], linewidths=1.2)
        ann = data.get(cat, {"arrows": [], "circles": []})
        for (x0, y0, x1, y1) in ann.get("arrows", []):
            axrow[1].annotate("", xy=(x1, y1), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, shrinkA=0, shrinkB=1))
        for (cx, cy, rad) in ann.get("circles", []):
            axrow[1].add_patch(Circle((cx, cy), rad, fill=False, edgecolor=RED, lw=1.6, ls="--"))
        axrow[1].set_axis_off()
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.03)
    for ext in ("pdf", "png", "svg"):
        p = f"{OUT_DIR}/figura_categorias.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02); print("saved", p)
    plt.close(fig)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "annotate"
    (annotate if mode == "annotate" else render)()
