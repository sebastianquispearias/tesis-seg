# Guion de la defensa

Defensa: 24 de septiembre de 2026, 14h. 20 a 30 minutos.
Banca: Alberto Barbosa Raposo (PUC-Rio), Geraldo Braz Júnior (UFMA), Cesar Augusto
Sierra Franco (Tecgraf). Orientador Paulo Ivson, coorientador Luiz Fernando Trindade
Santos.

**Alberto y César estuvieron en la propuesta y van a repreguntar sus propios temas.
Geraldo no estuvo y abre el trabajo por primera vez.** Los primeros slides son
para él.

Lo que va entre `>` es lo que dices en voz alta. Lo demás es para leer antes.

---

## ERRORES ENCONTRADOS AL AUDITAR LOS SLIDES CONTRA LA TESIS

Seis, por orden de gravedad. **Los seis ya están corregidos en los slides.** Se dejan
escritos porque son el tipo de error que vuelve si alguien regenera el deck desde el
de junio.

**1. El slide de respaldo 28 tiene tres hiperparámetros mal.** Es el que un examinador
cruza con la tesis.

    el slide dice           la tesis dice
    lr = 10^-4              lr = 10^-3
    Batch size 8            5 etiquetadas + 1 sin etiquetar
    Early stopping en Dice  early stopping en IoU de validación

**2. La portada dice "Defesa de Proposta de Mestrado" y "Junho de 2026".** Es la
defensa y es septiembre. Y el coorientador es Luiz Fernando **Trindade** Santos.

**3. El slide 10 llama "Dice entre anotadores" al 0,854 de UNM.** Ese número es el
acuerdo con la **especialista clínica**. UNM lo anotó una sola persona, así que no
tiene medida entre anotadores. La que sí existe es la de INCA, 0,874, sobre los 898
frames anotados por duplicado.

**4. El slide 2 generalizaba de más** sobre el tamaño del pool. Corregido en el guion.

**5. El slide 25 cita UniMatch**, que se quitó de la tesis. Quedan FixMatch y
cross-pseudo supervision.

**6. El slide 12 describe el umbral a medias.** La máscara de confianza retiene los
píxeles con p ≥ 0,95 **o** p ≤ 0,05, o sea también el fondo confiado.

---

## 1. Portada

No hables. Deja que lean el título mientras te presentan.

## 2. O que este trabalho pergunta, e o que encontrou

**Treinta segundos, y es el slide más importante para Geraldo.**

> Este trabajo hace tres preguntas sobre aprendizaje semi-supervisado y las tres
> tienen respuesta medida.
>
> La primera es en qué régimen de anotación ayuda, y la respuesta es que solo donde el
> supervisado todavía no ha saturado, es decir, donde todavía mejora si le das más
> etiquetas. La segunda es si importa el tamaño del conjunto
> de frames sin anotar, y sí importa, pero de forma distinta en cada dataset. Y la
> tercera es si conviene elegir esos frames por cercanía temporal, y la respuesta es
> que no.
>
> Y una cuarta cosa que no era una pregunta de partida: si la máscara mejor sirve para
> algo aguas abajo. Sirve para colocar una región de interés, y no sirve para el
> escalar anatómico, por una razón que veremos al final.

No desarrolles. Cada línea tiene su slide después.

## 3. Divisor: Introdução

Pasa.

## 4. Disfagia e VFSS

> La disfagia es la dificultad para tragar. En pacientes de cáncer de cabeza y cuello
> es frecuente, tanto por el tumor como por el tratamiento, y su complicación grave es
> la aspiración silenciosa: comida o líquido que entra en la vía aérea sin provocar
> tos, y que acaba en neumonía.
>
> El examen de referencia para estudiarla es la videofluoroscopía de la deglución. Es
> un vídeo de rayos X en vista lateral mientras el paciente traga un contraste.
>
> Y aquí entran las vértebras. C2 a C4 son la referencia anatómica del examen. La
> anatomía cambia de un paciente a otro, así que para comparar medidas hay que
> normalizarlas, y se normalizan tomando esas vértebras como escala. Hoy eso se hace a
> mano: el clínico marca los puntos en ImageJ, frame a frame. Automatizar esa medida es
> lo que motiva segmentar las vértebras, y por eso el trabajo va de segmentación y no
> de clasificar si hay aspiración o no.

**Si preguntan por qué segmentar y no detectar dos puntos:** porque hay tareas aguas
abajo que necesitan la extensión espacial, no solo dos coordenadas. Slide 23 y tarjeta
8 del banco.

## 5. O gargalo da anotação

**Aquí van los números, no el adjetivo.**

> El problema no es que anotar sea difícil. Es que no escala.
>
> Cada máscara lleva entre dos y tres minutos a un anotador ya entrenado. Los 325
> frames de UNM son de diez a dieciséis horas de trabajo, y los 968 de INCA, entre
> treinta y dos y cuarenta y ocho. Y eso sin contar la revisión de calidad: corregir
> una máscara marcada como problemática tiene una mediana de 224 segundos.
>
> Además la referencia no es exacta. En INCA anotamos 898 frames por duplicado, y el
> acuerdo entre los dos anotadores es de 0,874 de Dice. Ese número es en la práctica un
> techo: un modelo que superara ese acuerdo estaría ajustándose al estilo de un
> anotador concreto.
>
> Y mientras tanto el material está ahí. Cada vídeo son miles de frames ya grabados. En
> UNM hay 343 frames sin anotar por cada uno anotado, en INCA 49. En entrenamiento
> supervisado todo eso se descarta.

## 6. Semi-supervised learning

> El aprendizaje semi-supervisado usa los dos conjuntos a la vez: la pérdida
> supervisada sobre los frames anotados, y una segunda pérdida que no necesita
> etiquetas sobre todos los demás.
>
> La condición para que eso funcione es que los dos conjuntos vengan de la misma
> distribución. En la mayoría de los trabajos eso es una suposición que hay que
> argumentar. Aquí se cumple por construcción: los frames sin anotar salen de los
> mismos vídeos que los anotados, del mismo equipo, del mismo paciente y de la misma
> anatomía. La única diferencia entre un frame anotado y uno sin anotar es que a uno le
> tocó el muestreo y al otro no.

## 7. A lacuna de SSL em VFSS

> Busqué en Scopus el cruce de las dos familias de términos, semi-supervisado y
> videofluoroscopía, en junio. Diecinueve documentos.
>
> Revisándolos uno a uno, diecisiete no son de videofluoroscopía: son de páncreas, de
> muslo, de columna lumbar, de ecocardiografía. Los otros dos estudian masticación y
> deglución a partir de audio, no de imagen.
>
> Ninguno aplica aprendizaje semi-supervisado a la segmentación de vértebras en VFSS.
> Eso no quiere decir que el SSL sea nuevo en imagen médica, que es un campo enorme.
> Quiere decir que en este problema concreto no hay con qué compararse, y que las
> decisiones de diseño hay que tomarlas y justificarlas desde cero.

**No digas "somos los primeros en aplicar SSL a VFSS".** Di que no hay trabajos de SSL
para segmentación de vértebras en VFSS, que es lo que la búsqueda sostiene.

## 8. Perguntas de pesquisa

> Las tres preguntas, formalmente.
>
> La primera es sobre el régimen: en qué situación de escasez de etiquetas aporta el
> SSL algo que el supervisado no dé ya.
>
> La segunda y la tercera son sobre el conjunto sin anotar, y lo trato como una
> variable de diseño en vez de como algo dado: cuánto material hace falta, y si conviene elegirlo con
> algún criterio o da igual.

## 9. Divisor: Setup experimental

Pasa.

## 10. Datasets

> Dos datasets de dos instituciones.
>
> UNM es un subconjunto público de la base TalkBank: 35 vídeos de 35 pacientes, de
> población mixta. INCA viene del Instituto Nacional de Câncer, con 234 vídeos de 115
> pacientes tratados por cáncer de cabeza y cuello.
>
> Los elegí porque son estructuralmente distintos, y esa diferencia da dos regímenes.
> Los vídeos de UNM son largos, dos minutos de media, con varias degluciones, así que
> hay miles de frames por vídeo pero solo diez anotados. Los de INCA duran seis segundos
> y capturan una sola deglución.
>
> El resultado está en las dos últimas filas. INCA tiene el triple de etiquetas, y UNM
> tiene un pool sin anotar cuarenta veces mayor. Uno es el caso de pocas etiquetas y
> mucho material; el otro, el de más etiquetas y menos material.
>
> Los dos siguen el mismo protocolo de anotación y los dos se parten por paciente, para
> que no haya frames del mismo paciente en entrenamiento y en test.

**Cuidado con el Dice de anotación.** El 0,854 de UNM es el acuerdo con la
especialista, no entre anotadores.

## 11. Arquitetura de segmentação

> U-Net++ con un encoder EfficientNet-B3 preentrenado en ImageNet, a 320 por 320, y una
> pérdida que combina entropía cruzada y Dice.
>
> Pero lo que importa de este slide no es la arquitectura. Es la palabra fija.
>
> El backbone, la resolución, el optimizador, la augmentación de los frames anotados y
> todos los hiperparámetros generales se fijaron antes de la rejilla principal y no se
> tocaron en ningún experimento. Esa es la condición para que el resto del trabajo
> signifique algo: cuando dos condiciones dan resultados distintos, la diferencia no
> puede venir de haber ajustado una de las dos.
>
> El precio de fijarla es no saber si era la mejor elección. Por eso la comparé después
> contra otras cinco arquitecturas supervisadas, y eso es el slide 18.

**Si preguntan por qué U-Net++ y no una U-Net convencional:** la U-Net sale por encima,
0,824 contra 0,801, y la tesis lo admite. Se fijó antes de que esa comparación
existiera, por estabilidad entre semillas. Tarjeta 18 del banco.

## 12. Métodos avaliados: PL e MT

> Los dos métodos comparten el esquema profesor-alumno. El profesor no se entrena: sus
> pesos son una media móvil exponencial de los del alumno, con coeficiente 0,99. A cada
> frame sin anotar se le aplican dos augmentaciones, una débil que ve el profesor y una
> fuerte que ve el alumno.
>
> La diferencia entre los dos está en qué se le pide al alumno que iguale.
>
> Pseudo-etiquetado binariza la salida del profesor y se queda solo con los píxeles
> donde está confiado, por encima de 0,95 o por debajo de 0,05. El resto no contribuye
> a la pérdida.
>
> Mean Teacher no binariza. Persigue el mapa de probabilidad continuo, y ahí contribuyen
> todos los píxeles, incluidos aquellos donde el profesor no lo tiene claro.
>
> Esa diferencia parece un detalle de implementación y es la que explica el resultado
> de dentro de dos slides, así que la dejo dicha aquí.

## 13. Desenho experimental

> El estudio tiene tres ejes, y cada uno responde a una de las preguntas.
>
> El primero es el régimen de etiquetas. Reduzco el conjunto anotado a subconjuntos
> anidados: en UNM quitando frames dentro de cada vídeo, y en INCA quitando pacientes
> enteros, porque son dos estructuras distintas.
>
> El segundo es el tamaño del conjunto sin anotar. Lo construyo por radio temporal
> alrededor de cada frame anotado: radio tres son unos cien milisegundos a cada lado,
> radio veinte son casi setecientos. Y el extremo son todos los frames laterales del
> vídeo, 74.774 en UNM.
>
> El tercero es la política de selección. Comparo esa selección temporal contra un
> control aleatorio del mismo tamaño exacto, para que la única diferencia entre las dos
> condiciones sea el criterio y no la cantidad de material.

## 14. Divisor: Resultados

Pasa.

## 15. Resultado 1: quando o SSL ajuda

> Las dos curvas son la fracción de etiquetas contra el F1 en test.
>
> En UNM, el supervisado se queda en 0,801 y Mean Teacher llega a 0,860, cincuenta y
> nueve milésimas por encima. En INCA con el 10 % de las etiquetas, que son cincuenta y
> seis frames, sube de 0,844 a 0,865.
>
> Pero miren el extremo derecho de la curva de abajo. Con las 634 etiquetas completas,
> el supervisado ya está en 0,906 y el SSL no mueve nada.
>
> El patrón es el mismo en los tres casos y es el resultado principal del trabajo: el
> SSL aporta donde el supervisado todavía tiene margen, y no aporta donde ya ha
> saturado. En los regímenes evaluados la frontera cae alrededor de 0,85 de F1
> supervisado, aunque eso es lo observado aquí y no lo presento como una ley.

## 16. Resultado 2: o tamanho do pool (RQ2)

> En UNM los dos métodos reaccionan al contrario.
>
> Pseudo-etiquetado da su mejor resultado con el pool más pequeño, 0,852 con 1.257
> frames, y el peor con el más grande, 0,803, que es prácticamente el supervisado. Mean
> Teacher hace lo inverso: su mejor resultado es con los 74.774 frames.
>
> La explicación probable es la del slide de los métodos. Pseudo-etiquetado compromete
> una etiqueta dura por píxel, así que un error confiado del profesor se refuerza, y
> cuanto más grande es el pool más errores hay. Mean Teacher persigue un objetivo
> continuo y lo tolera mejor.
>
> Y hay que acotarlo, porque esto es de UNM. En INCA con todas las etiquetas el tamaño
> del pool no tiene ningún efecto discernible: los catorce brazos caen entre 0,902 y
> 0,908, todos dentro de cuatro milésimas del supervisado. Y en el subconjunto del 10 %
> el mejor pool se desplaza de sitio.
>
> Lo que sobrevive a los tres regímenes es esto: el mejor conjunto sin anotar depende
> del método y del dataset, y usar todo lo disponible no es la opción segura.

**Dilo tú antes de que te lo pregunten.** Es el punto donde el trabajo puede parecer que
generaliza de más.

## 17. Resultado 3: a proximidade temporal não ajuda (RQ3)

> Comparé la selección temporal contra un control aleatorio del mismo tamaño, en cada
> radio.
>
> Y lo importante no es que la diferencia sea pequeña. Es que el signo cambia. Temporal
> gana en radio tres y en radio siete, y pierde en radio diez y en radio quince. Eso es
> la firma del ruido, no la de un efecto.
>
> En el radio de referencia del trabajo, que es diez, los tests de Wilcoxon a nivel de
> paciente no detectan diferencia con ninguno de los dos métodos, en ninguna semilla. Y
> en INCA las diferencias entre las dos políticas quedan por debajo de una milésima en
> la mayoría de las configuraciones.
>
> Esperaba lo contrario. En vídeo, los frames vecinos parecían los candidatos naturales.
> No lo son, y una explicación plausible es que comparten anatomía con el frame que ya
> está anotado, así que aportan poca información nueva.

## 18. Resultado 4: SSL vs arquiteturas supervisionadas

> Con el pipeline fijo, las seis arquitecturas supervisadas van de 0,759 la peor a 0,851
> la mejor, que es TransUNet. U-Net++, la que uso en todo el trabajo, queda en 0,801, o
> sea ni la mejor ni la peor.
>
> Y las mejores configuraciones de SSL sobre U-Net++ llegan a 0,852 y 0,860, es decir al
> nivel de la mejor arquitectura supervisada del pipeline.
>
> Dicho de otro modo: usar los frames que ya están grabados da una ganancia del mismo
> orden que cambiar de arquitectura, sin cambiar de arquitectura.
>
> La última fila es nnU-Net, con 0,907, y no es comparable con las demás: trabaja a 1024
> y con su propio pipeline completo. Está aquí como referencia de contexto, y el slide
> siguiente pero uno va precisamente sobre esa diferencia.

## 19. SSL em outros backbones

> Para comprobar que esto no era una propiedad de U-Net++, repetí una sola configuración
> de Mean Teacher sobre los seis backbones.
>
> Cuatro de seis ganan, entre 0,012 y 0,029. Así que el efecto no está atado al modelo
> que elegí.
>
> Pero no llega a todos, y eso también hay que decirlo. BiFPN no se mueve. Y TransUNet,
> que es la mejor supervisada de las seis, pierde veinte milésimas.
>
> En la propuesta se sugirió lo contrario: que TransUNet, por ser más profunda, debería
> beneficiarse más del SSL. Lo medimos y sale al revés. La lectura que encaja con el
> resto del trabajo es la saturación: TransUNet parte de 0,851, que es justo donde el
> SSL deja de ayudar en todos los demás experimentos.

## 20. nnU-Net: o que explica a diferença?

> nnU-Net llega a 0,907, por encima de todo lo que entrené. Pero es un sistema
> autoconfigurable y su pipeline difiere del mío en cuatro cosas a la vez: resolución,
> intensidad de la augmentación, duración del entrenamiento y normalización. Comparar
> así no permite atribuir nada.
>
> Así que en vez de defender la diferencia, la descompuse. Cogí cada configuración de
> nnU-Net y la trasplanté sola a mi pipeline, dejando todo lo demás igual, y la medí
> contra su propio control con tres semillas.
>
> Entrenar a 1024 no cambia nada. La normalización por imagen tampoco. La augmentación
> moderada tampoco. Y la única consistente, con las tres semillas moviéndose en la misma
> dirección, es que la augmentación completa de nnU-Net empeora casi dos centésimas.
>
> O sea que ninguna de las diferencias explica por sí sola la ventaja. Lo que la explica
> es el conjunto del pipeline, no un ingrediente, y eso es una conclusión distinta de
> "nnU-Net es mejor".

**Esto responde a lo que Alberto y César plantearon en la propuesta.** Si preguntan por
480 de resolución: no la probamos, probamos la alternativa que estaba en cuestión.

## 21. Resultados qualitativos

> Tres columnas, tres regímenes, y lo mismo que las tablas pero visto.
>
> En UNM la predicción supervisada se deja parte de las vértebras y Mean Teacher las
> recupera: de 0,732 a 0,833. En el subconjunto del 10 % de INCA pasa lo mismo, de 0,770
> a 0,876.
>
> Y en INCA completo las dos predicciones son indistinguibles, 0,912 y 0,912. Ese es el
> resultado nulo de antes, visto en una imagen en vez de en una tabla.
>
> El rojo es lo que el modelo se dejó y el azul lo que puso de más.

Deja unos segundos de silencio. Es el único slide donde el efecto se ve.

## 22. Incerteza preditiva

> Esta parte salió de una sugerencia de la propuesta.
>
> Mido la incertidumbre como la entropía binaria de la probabilidad predicha, píxel a
> píxel, dentro de una banda de cinco píxeles alrededor del contorno de referencia. El
> máximo posible es 0,693. La banda da 0,049, dentro de las vértebras 0,029, y el fondo
> prácticamente cero. O sea que el modelo duda donde tiene que dudar, en el borde.
>
> El resultado es negativo y es el interesante: el SSL no deja al modelo más seguro. La
> incertidumbre total no baja, 0,049 contra 0,048. Lo que cambia es dónde está: baja
> dentro de las vértebras y sube en los huecos entre ellas.
>
> Y el tamaño del pool tampoco la mueve, que es lo que enseña el gráfico.
> Multiplicarlo por sesenta la deja igual. Cambiar la semilla la mueve más que
> multiplicar el pool por sesenta.

**NO digas que la incertidumbre sirve para seleccionar frames.** Lo probamos y el
contraste preregistrado sale nulo.

## 23. Consequências práticas

> Dos tareas aguas abajo, porque el Dice por sí solo no dice si esto sirve para algo.
>
> La primera es la región de interés dinámica que propone Lee para detectar invasión de
> la vía aérea. La ventana tiene tamaño fijo, así que la máscara no la redimensiona:
> solo la coloca. Mean Teacher deja el centroide a 4,62 píxeles del de referencia contra
> 6,09 del supervisado, y gana en 44 de los 63 frames del test.
>
> La segunda es el escalar anatómico C2-C4, y aquí la respuesta es incómoda y prefiero
> darla yo. La mejora en la máscara no se traslada al escalar. El error mediano baja de
> 9,3 a 7,9 píxeles y ahí se queda.
>
> El motivo es que la regla que localiza las esquinas de C2 y C4 tiene un suelo propio.
> Se lo apliqué a las máscaras de referencia, donde la segmentación es correcta por
> construcción, y aun así deja 6,9 píxeles de error mediano. Ese suelo se come el 87 %
> del error de Mean Teacher.
>
> Así que para ese escalar, más ganancia tiene que venir de la regla, o de predecir las
> esquinas directamente, no de mejores máscaras.

## 24. Contribuições

> Cuatro contribuciones.
>
> Una evaluación sistemática de SSL clásico para este problema, en dos datasets
> institucionales con regímenes de anotación distintos.
>
> Un estudio controlado de cómo se construye el conjunto sin anotar, tamaño y política,
> con el resto del pipeline fijo, que es lo que hace las diferencias atribuibles.
>
> Una comparación contra seis arquitecturas supervisadas y contra nnU-Net, con esa
> diferencia descompuesta configuración por configuración en vez de dejada como una
> brecha sin explicar.
>
> Y dos tareas aguas abajo medidas, no solo el Dice, que es lo que permite decir si esto
> sirve más allá de la métrica.

**No digas "primer trabajo que aplica SSL a VFSS".** Di "primera evaluación
sistemática", que es lo que la búsqueda del slide 7 sostiene.

## 25. Limitações

> Cuatro limitaciones, y prefiero decirlas yo.
>
> El análisis principal usa un solo backbone, así que los efectos podrían diferir con
> otros, aunque el slide de los seis backbones da alguna evidencia de que no del todo.
>
> Las familias más recientes de SSL no están probadas, aunque son refinamientos de las
> dos que sí, y el trabajo enseña que las dos se comportan casi igual entre ellas.
>
> El test de UNM son siete pacientes, lo que limita la potencia de los tests a nivel de
> paciente y obliga a leer los p-valores junto a los tamaños de efecto.
>
> Y las máscaras de referencia las hicieron estudiantes formados con criterios definidos
> con una fonoaudióloga, no clínicos anotando cada frame. El Dice mide acuerdo con ese
> criterio, no con una verdad absoluta.

## 26. Conclusões

> Tres conclusiones.
>
> El SSL clásico mejora la segmentación de vértebras cervicales en VFSS cuando el
> supervisado no ha saturado, y no cuando ya lo ha hecho. Eso acota dónde tiene sentido
> usarlo, que es lo contrario de decir que siempre ayuda.
>
> El mejor conjunto sin anotar depende del método y del dataset. Usar todos los frames
> disponibles no es la opción segura, y esa es la parte que alguien que quiera aplicar
> esto se lleva a casa.
>
> Y la proximidad temporal, que era la hipótesis intuitiva tratándose de vídeo, no aporta
> ventaja sobre elegir al azar el mismo número de frames.

## 27. Obrigado

---

## SLIDES DE RESPALDO, y a cuál saltar

No se enseñan. Están para responder, y hay que saber el número sin buscarlo.

    28   Detalhes do treinamento    lr, batch, early stopping, lambda_U, el umbral
                                    -> "¿cómo fijaste los hiperparámetros?" (tarjeta 18)
    29   Outras métricas, UNM full  IoU por imagen, en la misma convención que el F1
                                    -> "¿y el IoU?" o "¿precisión y recall?"
    30   Divisão dos datasets       partición por paciente, sin fuga temporal
                                    -> "¿hay fuga entre train y test?"
    31   O percurso de um frame     entrada, preproceso, red, umbral, métrica, y la
                                    rama SSL que solo existe en entrenamiento
                                    -> "entra un frame, ¿por dónde pasa?" (tarjeta 22)

El 31 es el que pide siempre quien abre el trabajo por primera vez. En la defensa de
Juliana fue la primera crítica del examinador externo, dicha así: *"eu peguei aqui
minha imagem de 384 por 384, daqui pra frente, ela passa por onde?"*. Geraldo Braz
llega en esa posición.

---

## LO QUE NO DEBE DECIRSE

- "Primer trabajo que aplica SSL a VFSS". Solo "primera evaluación sistemática".
- "El SSL siempre mejora". Solo donde el supervisado no ha saturado.
- "La incertidumbre sirve para seleccionar frames". El contraste preregistrado es nulo.
- "Los dos métodos van al contrario", sin decir que eso es de UNM.
- "No ajustamos hiperparámetros". Se comparó λ_U contra 0,10. Lo defendible es que la
  configuración es la misma en todas las condiciones.
- "Los resultados de la tesis eran un suelo". Se midió el brazo alineado y no sube.
