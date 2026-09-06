# Utiles de sesion

Scripts de lectura y verificacion que se escribieron en el scratchpad temporal
durante los dias 3 al 5 de septiembre y se guardaron aqui para no perderlos.
Ninguno entrena ni escribe en el `.tex`: todos son de solo lectura salvo el
ultimo, que anexa bloques a un archivo de texto.

| script | que hace |
|---|---|
| `resumen_runs.py` | media y SD de F1, ASSD y HD95 por brazo, leyendo los `run_report.json`. `python resumen_runs.py runs_X runs_Y` |
| `hubo_pool.py` | dice si un run SSL consumio datos sin etiquetar, mirando `max(unsup_loss)` del `epoch_history`. Es la comprobacion que destapo los 19 runs sin pool |
| `pareado.py` | compara dos brazos semilla a semilla, con delta medio, SD y coincidencia de signo |
| `comparar_cfg.py` | diferencias entre los `config` de dos runs, y su huella de reproducibilidad. Es como se vio que dos brazos se diferenciaban en un solo campo |
| `cola_nb.py` | busca un patron en las celdas de codigo de un notebook y las imprime con numero de celda y linea |
| `sacar_celda.py` | imprime una celda de un notebook, o un rango de lineas, en UTF-8 |
| `claves_historia.py` | claves del `epoch_history` y cuantas epocas traen valor no nulo. Sirve para no leer una clave que otra version renombro |
| `universo_pools.py` | comprueba que varios pools salgan del mismo universo, con la misma composicion por video y sin solapamiento |
| `anadir_bloque_estado.py` | anexa un bloque a `ESTADO_SESION.md` respetando sus finales de linea, con copia previa y comprobacion de reversibilidad |

## Por que importa el ultimo

`ESTADO_SESION.md` usa CRLF y el banco de defensa usa LF. Anexar con los finales
equivocados deja el archivo mezclado y no se nota hasta que algo se rompe. El
script detecta cual usa el destino y adapta el fragmento, y luego cuenta los CR y
LF sueltos para que quede constancia.
