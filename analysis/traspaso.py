"""Consolidated handoff: everything that only lived in the conversation."""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\ESTADO_SESION.md"
COPIA = RUTA + ".ANTES_traspaso_4sep"

BLOQUE = """

# ==================== 4/9 - TRASPASO CONSOLIDADO ====================

Lo que solo vivia en la conversacion. Con esto, un /clear no cuesta nada.

## COMMITS DE ESTOS DOS DIAS, repo tesis-seg, rama main

```
774ca78  copia del pool con 32 hilos, y candidatos elegidos antes de copiar
67ada93  el pool lateral se lee de un manifiesto, no listando la carpeta
1504918  notebook 22, pool por incertidumbre como tercera politica
87c4460  notebook 21, el control de resolucion que los runs a 1024 nunca dieron
3e69455  ARREGLO DEL RUIDO: restaura la intensidad pedida bajo albumentations 2.x
68c8816  notebook 20, Mean Teacher en los dos backbones que faltaban
9052350  el guardian cuenta las vistas aumentadas aplanadas
aec01b3  requirements sin los pines que ya no instalan
```

## LOS SCRIPTS, todos en tesis_seg/analysis/

```
EL ARREGLO DEL RUIDO Y SUS PUERTAS
  puerta_aug.py + aug_ANTES.json   los tres checks del arreglo. PUERTA ABIERTA.
  ruido_vs_senal.py                la demostracion: d' cae 94% con el ruido roto

EDICIONES DEL .tex (todas con --aplicar; sin el flag solo imprimen el antes/despues)
  reordenar_tabarch.py     el bloque SSL sigue el orden del supervisado
  recortar_pies.py         los dos pies recortados
  entradillas_45.py        las cinco marcas de 4.5 (v1)
  marcas_45_v2.py          de cinco a tres marcas (v2, la buena)
  partir_nnunet.py         parte el parrafo de nnU-Net
  borrar_horas_anotacion.py
  intro_42.py              la intro de 4.2 sin la contradiccion
  parear_controles.py      los dos controles de ImageNet con la misma forma
  arreglar_paradigma.py    la frase de cierre de nnU-Net
  definir_44.py            escalar, regla y "stray components"
  hd95_dispersion.py       los tres HD95 con su +/-
  etiquetar_44.py          cada cifra con su configuracion + la frase de la regla
  cerrar_roi.py            entradilla y cierre del parrafo del ROI
  partir_roi.py            parte el parrafo del ROI en dos
  aplicar_KR15.py          K_R 2.0 -> 1.5 y regenera la figura

PENDIENTE DE APLICAR
  editar_tex_backbones.py  las dos filas nuevas. TIENE PUERTA: no escribe si no
                           reproduce las 4 filas impresas desde los run_report.json.
                           HAY QUE ACTUALIZAR SU ANCLA por el reorden.
  filas_backbones.py       solo lectura: puerta + control de stack + las 2 filas

AUDITORIA Y MEDICION
  auditar_floats.py        23 flotantes, 0 huerfanos, 0 sin label
  revisar_estilo.py        pies, entradillas y muletillas
  leer_44.py               imprime una seccion frase a frase, en legible
  donde_se_define.py       donde se define cada termino en el documento
  bloques_arquitecturas.py
  que_dice_del_futuro.py   que dicen Limitations y la Conclusion
  medir_roi_frame.py       el desplazamiento del ancla en el frame de la figura
  probar_KR.py             renderiza la figura del ROI en tres escalas
  error_localization.py    area predicha, FN/FP, error al contorno (ya existia)
  c2c4_rule_floor.py       el suelo de la regla (ya existia)
  arreglar_cr.py           repara un CR suelto devolviendo la barra

NOTEBOOKS
  manifiesto_lateral.py    genera el manifiesto del pool lateral. CORRER EN LOCAL.
  verificar_nb22.py        chequeo estatico: nombres sin definir, procedencia de celdas
  probar_seleccion.py      prueba en seco la seleccion por video, sin leer imagenes
  fix_nb22_manifiesto.py / fix_nb22_candidatos.py   los dos parches del 22
```

## RESULTADOS MEDIDOS QUE NO ESTAN EN LA TESIS

```
Todos verificados leyendo los run_report.json o los CSV, nunca del .tex.

EL REGRID CONTRA nnU-NET          F1 cambia 0.0004 entre rejilla nativa y 320.
                                  ASSD y HD95 cambian 2-3x: por eso la tesis solo
                                  compara F1. Tarjeta 10 del wiki.
POR QUE r=15 EN LOS BACKBONES     r=10 es el mejor pool aleatorio (0.857) y r=15 da
                                  0.830. La eleccion SUBESTIMA la ganancia.
                                  Tarjeta 11 del wiki.
QUINTILES DEL ACTIVE LEARNING     INCA monotono (0.876 -> 0.769); UNM NO, y el area
                                  lo explica (rho +0.36 con entropia, -0.47 con Dice;
                                  en INCA -0.01 y -0.06).
EL SESGO DEL ESCALAR              +5.48 px con la MASCARA DE REFERENCIA, +6.72 el
                                  supervisado, +7.46 el MT. EL 82% LO PONE LA REGLA.
FN/FP Y ENCOGIMIENTO              area predicha 77.8% (sup) -> 88.6% (MT);
                                  FN/FP 4.81 -> 2.55. NO explica el escalar: el signo
                                  va al reves. Por eso no entro en la tesis.
LAS SEMILLAS DEL HD95 DE INCA     PL r20 [44.1, 22.2, 20.8], all-lateral
                                  [22.3, 50.5, 6.7], sup [8.0, 12.2, 21.8].
                                  La SD crece con el pool: 2.6, 5.2, 13.1, 22.2.
R = 299 DE LEE                    es el tamano de entrada de XCEPTION, no anatomia.
                                  Su caja ademas NO es cuadrada: el borde inferior va
                                  al fondo del frame.
LA VENTANA DEL ROI                K_R = 1.5 elegido mirando tres renders. Criterio:
                                  contener la columna con margen y dejar anatomia
                                  fuera. A 1.2 corta la C4.
```

## EL METODO QUE FUNCIONO, para seguir igual

```
1. Sebastian LEE el .tex y dice que le chirria. Cuatro de los quince arreglos salieron
   asi, y son los que ninguna auditoria automatica encuentra: la contradiccion de 4.2,
   la frase de las horas, el "learning paradigm", y los controles desparejos.
2. Antes de proponer, VERIFICAR contra los datos. Cayeron cinco ideas por esto:
   la figura de active learning, las horas de UNM, el sesgo, el HD95 vs pool, y la
   figura de backbones.
3. Antes de aplicar, ANTES/DESPUES y esperar el "dale".
4. Despues de aplicar, CR sueltos == 0, LF sueltos == 0, y compilar.
5. Cada script guarda copia .ANTES_* y muchos comprueban reversibilidad (que quitando
   el cambio el archivo vuelve a ser identico).
```
"""

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    c = fh.read()
if c.count("\r") != c.count("\r\n"):
    sys.exit("ABORTO: CR sueltos antes de tocar")
if "TRASPASO CONSOLIDADO" in c:
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
