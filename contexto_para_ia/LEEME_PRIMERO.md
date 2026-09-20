# Contexto de mi tesis, para un asistente de IA

## Quien soy y que necesito

Estudiante de maestria en informatica, PUC-Rio. Defensa el 24/09/2026.
Tesis: aprendizaje semi-supervisado para segmentacion de vertebras cervicales
C2-C4 en videofluoroscopia de la deglucion (VFSS). Imagenes 2D en gris (PNG).

**Mi objetivo es ENTENDER mi propio trabajo para defenderlo**, no escribir mas
codigo. Preguntame cosas y corrigeme cuando me equivoque.

## Como quiero que me expliques

- Concreto antes que abstracto: primero la cosa, despues su nombre tecnico.
- Un concepto por vez, con la historia completa, no resumenes escuetos.
- Leo en el celular: lineas CORTAS y angostas, nada de tablas anchas.
- Todo lo que afirmes tiene que poder verificarse en estos archivos.
- Si algo no esta aqui, decime que no lo podes verificar. No supongas.

## LO QUE NO TE PUEDO DAR, Y POR QUE

Las imagenes y las mascaras son de pacientes (INCA, y UNM TalkBank) bajo
convenio. **No se comparten.** Por eso los notebooks van SIN salidas y de las
carpetas de resultados solo vienen los numeros. Si necesitas saber algo de las
imagenes, preguntame y te lo describo.

## PROTOCOLO DE DATOS. Leelo antes de sacar cuentas

Mis frames NO estan repartidos al azar. Estan repartidos **por video**, para
que frames casi identicos del mismo video no caigan en entrenamiento y en
prueba a la vez.

```
35 videos en total

train   23 videos   218 frames etiquetados
val      5 videos    44 frames
test     7 videos    63 frames
                    ---
                    325 frames etiquetados
```

nnU-Net junta train + val en una sola carpeta `imagesTr`, y por eso mi
`dataset.json` declara:

```
numTraining: 262   =   218 (train) + 44 (val)
```

Los 63 de test van aparte, en `imagesTs`. **No falta ningun frame**:
262 + 63 = 325.

### El archivo 16_splits_final_OBSOLETO.json

Ese archivo dice 66 de entrenamiento y 44 de validacion. **NO es el protocolo
del baseline.** Es lo que quedo en disco despues de un experimento posterior de
fracciones de etiquetas: 66 es el 25% de 218. Las celdas de esa ablacion
sobrescribieron el archivo y no se restauro.

El baseline que reporto en la tesis entreno con 218, y el log de la corrida lo
dice literalmente:

```
This split has 218 training and 44 validation cases.
```

Si sacas cuentas con el 66, te van a dar mal. Lo incluyo solo para que veas el
archivo real y no supongas.

## Que es cada archivo

```
00_resumen_del_proyecto.md    el contexto entero: datos, metodos,
                              experimentos, decisiones ya tomadas
01_resultados_canonicos.md    los numeros buenos. Mandan sobre
                              cualquier otro archivo
02_analisis_readme.md         como se calcularon las metricas de borde

10_nnUNetPlans.json           el plan que nnU-Net decidio para mi dataset
11_dataset.json               como declare mi dataset a nnU-Net
12_dataset_fingerprint.json   el censo que nnU-Net midio de mis imagenes
13_nnunet_metrics.json        nnU-Net al 100% de etiquetas
14_nnunet_label_efficiency    nnU-Net al 25 / 50 / 75 / 100%
15_nnunet_regrid320.csv       nnU-Net reevaluado en la rejilla 320x320,
                              para compararlo con mi pipeline
16_splits_final_OBSOLETO      OJO: ver arriba

20..29c   mi pipeline: defaults, el bucle de entrenamiento,
          las perdidas, los pools, la aumentacion, los modelos,
          las metricas, la EMA, la evaluacion, el preprocesado,
          el BiFPN-UNet de Kim, y mi trainer de Mean Teacher
          (que hereda de nnUNetTrainer y va DENTRO de nnU-Net)

30_notebooks_sin_salidas/     los 25 notebooks: aqui esta lo que se
                              hizo de verdad, experimento por experimento.
                              Los titulos de los notebooks 16 a 22 son
                              literalmente las preguntas que contestan
40_scripts_del_metodo/        los 16 scripts que construyen los pools
                              sin etiqueta, las fracciones de etiquetas,
                              los tests de Wilcoxon y la reevaluacion de
                              nnU-Net en 320
50_todos_los_runs.csv         **una fila por cada uno de los 361 runs**,
                              con su configuracion y sus metricas de test.
                              Si quieres comparar cualquier cosa, empieza aqui

90_dissertacao.tex            el texto de la tesis
91_slides.tex                 los slides de la defensa
92_guion_de_la_defensa.md     el guion hablado
93_preguntas_que_me_pueden    20 preguntas que me pueden hacer.
                              Usalas para interrogarme: hazme UNA,
                              espera mi respuesta, y corrigeme
94_paper.tex                  el paper enviado a CMPB
```

## Cosas verificadas que conviene que sepas

- **20_defaults.py NO es lo que se uso.** Medido sobre los 364 config.json de
  los runs finales: `patience_es` fue 40 en los 364 (el default dice 20), y
  `semi_start_epoch` fue 15 en UNM y 7 en INCA con SSL activo (el default dice
  30, que solo aparece en los supervisados, donde no hace nada). Para saber que
  uso cada run, mira `50_todos_los_runs.csv`, no `20_defaults.py`.

- **Constantes del estudio** (un unico valor en los 364 runs): entrada 320x320,
  lote 5, lr 1e-3, weight_decay 1e-4, warmup 10, BCE+Dice, umbral 0.5,
  tau 0.95, ema 0.99, preproc "base", rotacion 5, escala 0.03, sin flip.
  **Variado a proposito:** metodo SSL, con/sin SSL, arquitectura, lambda_u.

- **Sobre nnU-Net:** no se modifico su codigo. Lo que se hizo fue
  (a) convertir el dataset a su formato, (b) reemplazar su validacion cruzada
  aleatoria por mi division POR VIDEO, (c) entrenar **un solo fold, sin
  ensemble**, y (d) una sola corrida por fraccion. No se corrio
  `find_best_configuration`, asi que no hay postprocesado ni seleccion de
  configuracion: las predicciones de nnU-Net son crudas, igual que las mias.

- **El plan que nnU-Net eligio:** parche 1024x1024, lote 3, PlainConvUNet de
  9 etapas, ZScoreNormalization, solo configuracion 2d. Mi imagen mediana es
  898x968, asi que el parche es MAS GRANDE que 166 de mis 262 imagenes: esas se
  rellenan, no se recortan, y el sobremuestreo de primer plano casi no actua.

- **Mis imagenes no tienen calibracion fisica.** Las 262 declaran espaciado
  [1, 1], el valor por defecto del conversor. Medido sobre las mascaras, la
  columna cervical ocupa entre 147 y 481 pixeles segun el video (factor 3.27),
  y nada corrige esa diferencia: la absorbe la aumentacion de escala, no el
  preprocesado.
