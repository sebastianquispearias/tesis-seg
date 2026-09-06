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
| `pareado_generico.py` | comparacion pareada entre DOS carpetas cualesquiera, con la cola de `unsup_loss`. `pareado.py` lleva sus comparaciones cableadas e ignora argv; este no |
| `inventario.py` | por cada carpeta de semilla: si esta completa, cuantas epocas, que artefacto falta y cuando se escribio por ultima vez. Distingue un run cortado de uno que sigue entrenando |
| `auditoria.py` | tabla de TODOS los brazos del proyecto con n, F1, dataset, n_img, `use_semi` y `lambda_u`. Sirve para ver de un vistazo que brazos no son comparables |
| `desajuste_nombres.py` | carpetas cuyo informe lleva un `experiment_name` distinto del nombre de la carpeta, y las medias recalculadas agrupando por carpeta |
| `verificar_tabarch.py` | recalcula las 14 filas de `tab:arch` desde los informes y las compara digito a digito con lo impreso en el `.tex` |
| `huellas_tabarch.py` | version de albumentations, torch, smp y python de cada semilla de `tab:arch`, para ver si una fila mezcla entornos |
| `pesos_cobertura.py` | empareja cada semilla sin `best_model.pt` con su fichero en `PESOS_UNM` o `PESOS_INCA` |
| `replicas_aug_v2.py` | compara los perfiles de augmentation dejando fuera el brazo entrenado con el ruido roto |

## Por que importa el ultimo

`ESTADO_SESION.md` usa CRLF y el banco de defensa usa LF. Anexar con los finales
equivocados deja el archivo mezclado y no se nota hasta que algo se rompe. El
script detecta cual usa el destino y adapta el fragmento, y luego cuenta los CR y
LF sueltos para que quede constancia.


## Las dos trampas que motivaron los scripts del 6/9

`resumen_runs.py` agrupa por el campo `experiment_name` del informe, no por la
carpeta. Dos informes de `runs_final_v1` llevan un nombre acabado en `_debug` que
no coincide con su carpeta, asi que la herramienta parte el brazo y dice `n=2`
donde hay tres semillas. Una de esas carpetas es la que produce el `.851` de
`tab:arch`. Antes de creer que falta una semilla, comprobar con
`desajuste_nombres.py`.

Dos carpetas cuyo `config` coincide no son dos replicas si el codigo cambio entre
una y otra. `comparar_cfg.py` compara la configuracion, no el codigo.
`A_baseline` y `A_baseline_fixnoise` difieren en un solo campo, `exp_dir`, y sin
embargo la primera se entreno con el ruido gaussiano desactivado por
albumentations 2.x. Promediarlas mezcla un defecto con su arreglo.
