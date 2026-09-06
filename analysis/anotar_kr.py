"""Record the window scale decision for the region-of-interest figure."""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\ESTADO_SESION.md"
COPIA = RUTA + ".ANTES_bloque_kr"

BLOQUE = """

# ======= 4/9 - LA VENTANA DE LA FIGURA DEL ROI: K_R 2.0 -> 1.5 =======

Sebastian dijo que el cuadrado se veia muy grande y que no parecia aislar nada. Tenia
razon, y al buscarle fundamento al tamano resulto que NO LO TENIA.

## POR QUE EL TAMANO ERA LIBRE

```
R = 299 en Lee              ES EL TAMANO DE ENTRADA DE XCEPTION, no anatomia. El propio
                            paper lo delata: "299x299x3  19x19x728 feature maps".
                            Copiar ese numero a otra geometria no significa nada.
"bottom of the image"       Lee ancla el borde INFERIOR de su ROI al fondo del frame.
                            Eso NO transfiere aqui: nuestros frames van rellenados a
                            cuadrado (pad_sq), asi que su fondo es relleno negro.
la proporcion 0.7 / 0.3     ESA SI es de Lee, y es la unica que el pie reclama.
K_R = 2.0                   estaba en el script con un comentario y sin derivacion.
```

## LA CITA EXACTA DE LEE, por si hace falta

```
"the coordinates of the upper left and upper right corners are determined heuristically
 based on (mx, my) to be (mx - 0.7R, my - 0.5R) and (mx + 0.3R, my - 0.5R),
 respectively, where R = 299"
"The BOTTOM of the ROI is set to coincide with the BOTTOM OF THE IMAGE"
"(C) ROI image with dimensions of 299 x 299"
Fuente: papers_review/bibliografia txt/A_VFSS_Cervical_Vertebra_and_Anatomy/
        A7_Lee_Automatic_detection_airway_invasion_videofluoroscopy_deep_learning.txt
OJO: LA CAJA DE LEE NO ES UN CUADRADO. Solo su borde superior depende del centroide.
La nuestra si lo es, y el pie NO afirma lo contrario: reclama solo las proporciones.
```

## EL CRITERIO CON EL QUE SE ELIGIO 1.5

```
Se renderizo la figura en tres escalas y se miraron (scratchpad/probar_KR.py, que
escribe fuera de figs/ para no tocar la oficial):

  K_R = 2.0   R = 194 px  61% del frame   casi nada queda fuera; parece un recuadro
  K_R = 1.5   R = 146 px  45%             contiene las tres vertebras con margen y
                                          deja mandibula y cuello visibles fuera
  K_R = 1.2   R = 116 px  36%             CORTA LA C4 en el panel del supervisado

CRITERIO: la caja debe CONTENER la columna de referencia entera con margen, y dejar
anatomia VISIBLE fuera para que se lea como una seleccion. 1.5 es el unico que cumple
las dos.
```

## LO APLICADO

```
paper_figures/fig_roi_lee.py     K_R = 2.0 -> 1.5     copia .ANTES_KR15
figura regenerada en tesis_latex_overleaf/figs/ y tesis_figuras/ (pdf y png)
pie del .tex: "twice the height of the reference column"
           -> "one and a half times the height of the reference column"
                                                       copia .ANTES_KR15
paginas=52 errores=0 citas/refs sin resolver=0

NINGUN NUMERO MEDIDO CAMBIO:
  Supervised    offset 14.69 px   antes 7.6% del ancho, ahora 10.1%
  Mean Teacher  offset  2.25 px   antes 1.2%,           ahora  1.5%
Lo unico que cambio es contra que se comparan visualmente.
```

## LO QUE SE EVALUO Y SE DESCARTO

```
Rehacer la caja con la geometria REAL de Lee (borde inferior en el fondo del frame).
SE DESCARTO: haria FALSA la frase del texto "That window has a fixed size, so the
segmentation does not resize it, only moves it", porque con la geometria de Lee el
alto SI depende de donde caiga el ancla. Habria que reescribir texto, pie y la frase
que justifica por que se mide el centroide.
```

## PARA LA TARJETA DE DEFENSA

```
Si preguntan "por que esa ventana y no otra?":
  "La escala es ilustrativa y el pie lo dice. Lo que se reproduce de Lee son las
   proporciones, 0.7 anterior y 0.3 posterior, y el anclaje en el centroide. El
   tamano no se puede copiar: su R = 299 es el tamano de entrada de Xception, la red
   que ellos alimentan con el recorte, no una medida anatomica."
```
"""

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    c = fh.read()
if c.count("\r") != c.count("\r\n"):
    sys.exit("ABORTO: CR sueltos antes de tocar")
if "K_R 2.0 -> 1.5" in c:
    sys.exit("ABORTO: ya esta")
a = c.count("\r\n")
if not os.path.exists(COPIA):
    shutil.copy2(RUTA, COPIA)
with io.open(RUTA, "w", encoding="utf-8", newline="") as fh:
    fh.write(c + BLOQUE.replace("\r\n", "\n").replace("\n", "\r\n"))
with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    e = fh.read()
print("CR sueltos:", e.count("\r") - e.count("\r\n"))
print("LF sueltos:", e.count("\n") - e.count("\r\n"))
print("lineas    :", a, "->", e.count("\r\n"))
