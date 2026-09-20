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
- Todo lo que afirmes tiene que poder verificarse en los archivos adjuntos.
- Si algo no esta en los archivos, decime que no lo podes verificar.

## LO QUE NO TE PUEDO DAR, Y POR QUE

Las imagenes y las mascaras son de pacientes (INCA, y UNM TalkBank) bajo
convenio. **No se comparten.** Por eso aqui solo hay codigo, configuraciones y
numeros agregados. Si necesitas saber algo de las imagenes, preguntame y te lo
describo.

## Que es cada archivo

```
00_resumen_del_proyecto.md    el contexto entero: datos, metodos,
                              experimentos, decisiones ya tomadas
01_resultados_canonicos.md    los numeros buenos. Los de este archivo
                              mandan sobre cualquier otro
02_analisis_readme.md         como se calcularon las metricas de borde

10_nnUNetPlans.json           el plan que nnU-Net decidio para mi dataset
11_dataset.json               como declare mi dataset a nnU-Net
12_dataset_fingerprint.json   el censo que nnU-Net midio de mis imagenes
13_nnunet_metrics.json        resultado de nnU-Net al 100% de etiquetas
14_nnunet_label_efficiency    nnU-Net al 25 / 50 / 75 / 100%
15_nnunet_regrid320.csv       nnU-Net reevaluado en la rejilla 320x320,
                              para compararlo con mi pipeline
16_splits_final_OBSOLETO      ESTE ARCHIVO ESTA VIEJO. Ver abajo,
                              en PROTOCOLO DE DATOS

20_defaults.py                los valores por defecto. OJO: los notebooks
                              los sobrescriben (ver abajo)
21_train.py                   el bucle de entrenamiento: supervisado,
                              pseudo-label y Mean Teacher
22_losses.py                  BCE + Dice, y la perdida sin etiqueta
23_datasets.py                los pools de frames sin etiqueta
24_augmentations.py           aumentacion suave, debil y fuerte
25_models.py                  la fabrica de modelos
26_metrics.py                 F1, IoU, BF1, ASSD, HD95
27_utils.py                   la EMA del profesor
28_nnUNetTrainerMeanTeacher   mi trainer propio: Mean Teacher DENTRO
                              de nnU-Net, heredando de nnUNetTrainer

90_dissertacao.tex            el texto de la tesis
91_slides.tex                 los slides de la defensa
92_guion_de_la_defensa.md     el guion hablado de la presentacion
93_preguntas_que_me_pueden    20 preguntas que me pueden hacer.
                              Usalas para interrogarme: hazme UNA,
                              espera mi respuesta, y corrigeme
```

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

Los 63 de test van aparte, en `imagesTs`. **No falta ningun frame**: 262 + 63
= 325.

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

## Cosas verificadas que conviene que sepas

- **defaults.py NO es lo que se uso.** Medido sobre los 364 config.json de los
  runs finales: `patience_es` fue 40 en los 364 (el default dice 20), y
  `semi_start_epoch` fue 15 en UNM y 7 en INCA con SSL activo (el default dice
  30, que solo aparece en los supervisados, donde no hace nada).

- **Constantes del estudio** (un unico valor en los 364 runs): entrada 320x320,
  lote 5, lr 1e-3, weight_decay 1e-4, warmup 10, BCE+Dice, umbral 0.5,
  tau 0.95, ema 0.99, preproc "base", rotacion 5, escala 0.03, sin flip.
  **Variado a proposito:** metodo SSL, con/sin SSL, arquitectura, lambda_u.

- **Sobre nnU-Net:** no se modifico su codigo. Lo que se hizo fue
  (a) convertir el dataset a su formato, (b) reemplazar su validacion cruzada
  aleatoria por mi division POR VIDEO, para que frames del mismo video no
  cayeran en train y val, (c) entrenar **un solo fold, sin ensemble**, y
  (d) una sola corrida por fraccion. No se corrio `find_best_configuration`,
  asi que no hay postprocesado ni seleccion de configuracion: las predicciones
  de nnU-Net son crudas, igual que las de mi pipeline.

- **El plan que nnU-Net eligio para mi dataset:** parche 1024x1024, lote 3,
  PlainConvUNet de 9 etapas, ZScoreNormalization, solo configuracion 2d.
  Mi imagen mediana es 898x968, asi que el parche es MAS GRANDE que 166 de mis
  262 imagenes: esas se rellenan, no se recortan.

- **Mis imagenes no tienen calibracion fisica.** Las 262 declaran espaciado
  [1, 1], que es el valor por defecto del conversor. Medido sobre las mascaras,
  la columna cervical ocupa entre 147 y 481 pixeles segun el video (factor
  3.27), y nada corrige esa diferencia.
