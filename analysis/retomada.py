"""Write the pick-up-here block at the end of the session state file."""

import io
import os
import shutil
import sys

RUTA = r"G:\My Drive\UNM_vertebras_seg_v3\ESTADO_SESION.md"
COPIA = RUTA + ".ANTES_retomada_4sep"

BLOQUE = """

# ==================== 4/9 - PUNTO DE RETOMADA ====================

Lo unico que hace falta leer para continuar. Todo lo de arriba es detalle.

## LA TESIS

```
tesis_latex_overleaf/tesis_final_v54.pdf   52 PAGINAS (ya no 53), 0 errores, 0 refs
compilar: bash tesis_latex_overleaf/compilar.sh
Quince ediciones el 3 y 4 de septiembre. NINGUNA toca una cifra de resultados.
Secciones repasadas frase a frase: 4.2, 4.3, 4.4, 4.5.
SIN REPASAR: 4.1, 4.6, 4.7. En la auditoria de estructura salieron sanas.
```

## LO QUE ESTA CORRIENDO O LISTO

```
nb 20  TransUNet y BiFPN con MT. Lanzado el 3/9 con el codigo arreglado.
nb 19  A_baseline_fixnoise, 3 semillas. Iba por 2 de 3.
nb 21  resolucion 320 vs 1024. ESCRITO Y PUSHEADO (87c4460), SIN LANZAR. ~11 h.
nb 22  pool por incertidumbre. Corriendo desde la noche del 4/9. Commit 774ca78.
       Copia con 32 hilos (~46 arch/s), no con tar (2 arch/s).
```

## LA UNICA EDICION VIVA DEL .tex

```
Las dos filas de backbones en tab:arch, esperando el notebook 20.
OJO: por el reorden del bloque, TransUNet va AL FINAL y BiFPN-U-Net(T) va SEGUNDA.
Hay que actualizar tesis_seg/analysis/editar_tex_backbones.py para que reemplace el
BLOQUE ENTERO de 4 filas por uno de 6, porque su ancla actual ya no vale.
El script tiene una puerta: no escribe si no reproduce las 4 filas impresas desde los
run_report.json. Probado: las reproduce.
```

## OFRECIDO Y SIN DECIDIR

```
1. "Figure 4.6 shows one case:" -> "In one case ... (Figure 4.6)."
   Es el ultimo dos puntos de prosa de 4.4.
2. Donde va el motivo de que no se evalue la deteccion de invasion (UNM no tiene
   etiquetas de bolo ni de invasion): tarjeta de defensa, o una frase en Limitaciones.
3. Si P3 de 4.3 pide corte cuando crezca con las filas de backbones.
4. Aclarar que queria decir con "hacer SSL al residuo".
```

## PENDIENTE

```
- Mensaje a Ivson: REDACTADO Y SIN ENVIAR. Decision suya, espera resultados.
- Avisar a Paulo del 0.808 -> 0.800 +/- 0.014.
- Las cuatro tarjetas de defensa (FASE 3), mas las que salieron estos dos dias:
  el regrid (0.0004), los quintiles del active learning, el sesgo de la regla
  (+5.5 px), r=15 en los backbones, y la escala de la ventana del ROI.
```

## LECCIONES DE ESTOS DOS DIAS, para no repetirlas

```
1. NUNCA pasar por bash texto con barras, backticks o parentesis. Ni heredoc ni
   python -c en linea. SCRIPT CON Write Y EJECUTAR POR RUTA. Fallo tres veces el 3 y
   4 de septiembre, de tres formas distintas, y una rompio este mismo archivo.
2. Comprobar CR sueltos == 0 y LF sueltos == 0 despues de tocar cualquier archivo.
   Eso detecto el fallo del punto 1.
3. NO EXTRAPOLAR el coste de una operacion desde otra con patron de acceso distinto.
   La Puerta 6 leia salteado y daba 79 h; el tar en orden daba 9.8 h; 32 hilos daban
   0.5 h. Medir la operacion que se va a usar.
4. EL CONTEXTO NO LO PUEDO MEDIR. El contador que ve el modelo es por turno, no la
   ventana. El dato real lo da /context y hay que PEDIRLO, no estimarlo. El 4/9 se
   estimo 30% cuando el real era 93%.
5. Las salidas de Bash pesan: 150k tokens de esta sesion (15% de la ventana). Filtrar
   con head/tail/grep en vez de volcar archivos enteros.
```
"""

with io.open(RUTA, "r", encoding="utf-8", newline="") as fh:
    c = fh.read()
if c.count("\r") != c.count("\r\n"):
    sys.exit("ABORTO: CR sueltos antes de tocar")
if "4/9 - PUNTO DE RETOMADA" in c:
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
