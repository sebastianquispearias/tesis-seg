# Resultados finales consolidados (VERIFICADO — auditoría de seeds)

Fuentes oficiales: `runs_final_v1/` (UNM), `runs_inca_final_v1/` (INCA),
`nnunet_baseline_results/*.json` (nnU-Net). Métrica principal: F1/Dice medio por
imagen en test (`sample_mean_f1`). Boundary: ASSD y HD95 en píxeles
(`test_boundary_metrics` de cada run_report; nnU-Net desde `boundary_metrics_run0.csv`).

**Conteo de seeds (verificado leyendo los run_reports):**
- **5 seeds:** arquitecturas supervisadas; r10 (PL/MT, temporal/random); fracciones de etiquetas UNM.
- **3 seeds:** resto de SSL UNM (r3/5/7/15/20 + all-lateral); todo INCA.
- **1 seed (single-run):** MT DeepLabV3+ r15.
- **single-run:** todo nnU-Net.

> Nota: una extracción previa a 3 seeds daba números incorrectos para arquitecturas,
> r10 y fracciones. Los valores de abajo son los correctos y coinciden con
> `tesis_final_v28.tex`.

---

## 1. UNM — Arquitecturas supervisadas (5 seeds), 320×320

| Modelo | F1 | ASSD (px) | HD95 (px) |
|---|---|---|---|
| TransUNet | .851 ± .011 | 1.92 ± 0.20 | 8.16 ± 1.49 |
| U-Net | .824 ± .032 | 2.09 ± 0.31 | 8.39 ± 2.22 |
| U-Net++ (ImageNet) | .801 ± .014 | 2.41 ± 0.22 | 9.86 ± 2.06 |
| FPN | .799 ± .025 | 2.08 ± 0.20 | 6.55 ± 0.87 |
| DeepLabV3+ | .776 ± .041 | 2.08 ± 0.23 | 7.07 ± 0.66 |
| Kim BiFPN-U-Net(T) (random init) | .759 ± .012 | 4.86 ± 1.59 | 12.39 ± 2.78 |
| Kim BiFPN-U-Net(T) (pretrained) | .762 ± .014 | 4.34 ± 0.95 | 12.18 ± 1.48 |
| nnU-Net (single-run, 1024²) | .907 | 2.52 | 9.53 |

## 2. UNM — ImageNet vs from-scratch (U-Net++)

| | F1 | seeds |
|---|---|---|
| ImageNet | .8015 ± .0140 | 5 |
| from-scratch | .7765 ± .0239 | 3 |

Diferencia = **+0.025** (ImageNet). Seed mismatch (5 vs 3). No está en la tesis v28.

## 3. UNM — Pseudo-Label (3 seeds; r10 = 5 seeds)

| Pool | Temporal | Random |
|---|---|---|
| r3 | .852 ± .019 | .835 ± .005 |
| r5 | .831 ± .008 | .826 ± .005 |
| r7 | .830 ± .019 | .819 ± .027 |
| r10 | .832 ± .012 | .845 ± .010 |
| r15 | .840 ± .007 | .846 ± .016 |
| r20 | .803 ± .022 | .806 ± .042 |
| all-lateral | .830 ± .020 | — |

## 4. UNM — Mean Teacher (3 seeds; r10 = 5 seeds)

| Pool | Temporal | Random |
|---|---|---|
| r3 | .828 ± .011 | .811 ± .023 |
| r5 | .837 ± .026 | .836 ± .008 |
| r7 | .834 ± .025 | .818 ± .030 |
| r10 | .850 ± .006 | .857 ± .004 |
| r15 | .828 ± .006 | .830 ± .038 |
| r20 | .828 ± .019 | .821 ± .026 |
| all-lateral | .860 ± .006 | — |

Supervisado U-Net++ = .801 ± .014 (5 seeds). Mejor SSL: MT all-lateral .860, MT random r10 .857.

## 5. UNM — Eficiencia de etiquetas (5 seeds)

| Fracción | Supervisado | PL r10 temporal | MT r10 temporal |
|---|---|---|---|
| 25% | .807 ± .010 | .803 ± .023 | .816 ± .017 |
| 50% | .823 ± .017 | .819 ± .026 | .819 ± .011 |
| 75% | .829 ± .017 | .834 ± .021 | .837 ± .015 |
| 100% | .801 ± .014 | .832 ± .012 | .850 ± .007 |

## 6. UNM — SSL en otros modelos (PILOTOS, MT random r15)

| Modelo | Supervisado (5s) | MT r15 | Δ | seeds MT |
|---|---|---|---|---|
| U-Net | .824 ± .032 | .851 ± .005 | +0.027 | 3 |
| FPN | .799 ± .025 | .811 ± .028 | +0.012 | 3 |
| DeepLabV3+ | .776 ± .041 | .808 | +0.032 | 1 (single-run) |

## 7. INCA — Full-label (3 seeds)

| | F1 | pool/política |
|---|---|---|
| Supervisado 100% (634 fr.) | .906 ± .003 | — |
| from-scratch | .905 ± .004 | — |
| Mejor PL | .909 ± .002 | r20 temporal |
| Mejor MT | .908 ± .001 | r3 temporal (y r7 random) |

INCA 100% saturado: todo PL/MT en .903–.909.

## 8. INCA — Eficiencia de etiquetas (patient fractions, 3 seeds)

| Fracción | Supervisado | PL r10 | MT r10 | Δ mejor SSL |
|---|---|---|---|---|
| 10% (56 fr.) | .844 ± .006 | .854 ± .006 | .865 ± .005 | +0.021 (MT) |
| 25% (132 fr.) | .892 ± .002 | .887 ± .003 | .888 ± .005 | −0.004 |
| 50% (283 fr.) | .896 ± .004 | .894 ± .000 | .897 ± .003 | +0.001 |

INCA patient10 (no saturado): MT ayuda +2.1 pts. Al 25%/50% ya casi saturado → SSL plano.

## 9. nnU-Net UNM (single-run, 1024²)

| Condición | F1 |
|---|---|
| Supervisado 25% | .896 |
| Supervisado 50% | .903 |
| Supervisado 75% | .908 |
| Supervisado 100% | .907 |
| Self-training offline 25% | .901 (+0.5 vs sup 25%) |
| Self-training offline 100% | .912 (+0.5 vs sup 100%) |
| Mean Teacher online 25% (150 ep) | .868 |
| Supervisado matched 25% (150 ep) | .863 |

⚠️ El MT online (150 ep) se compara SOLO contra su baseline matcheado (.863 → Δ +0.004),
NO contra el nnU-Net full 25% (.896).

---

## 10. Controles con `lambda_u = 0` — ¿de dónde viene realmente la mejora del SSL?

> **Estos controles estaban corridos desde antes y no figuraban en este archivo.**
> Fuente: `runs_control_lambda0/`, notebook `16_unm_control_lambda0.ipynb`.
> Verificado que la configuración es **idéntica a su run SSL salvo `lambda_u`**
> (comparación clave por clave; la única otra diferencia es `exp_dir`).

Un run con `use_semi=True` y `lambda_u=0` mantiene todo el régimen —profesor EMA,
aumentaciones débil/fuerte, calendario— pero la pérdida no supervisada pesa cero.
Aísla, por tanto, cuánto aporta el régimen y cuánto la pérdida de consistencia.

| Escenario | Supervisado | Control λ=0 | SSL λ=0.05 |
|---|---|---|---|
| UNM r10 | .8015 ± .0140 (5) | **.8410 ± .0144 (5)** | .8502 ± .0065 (5) |
| UNM all-lateral | .8015 ± .0140 (5) | **.8613 ± .0023 (3)** | .8602 ± .0058 (3) |
| INCA patient10 | .8442 ± .0064 (3) | **.8618 ± .0056 (4)** | .8654 ± .0048 (3) |

Descomposición de la ganancia sobre el supervisado:

| Escenario | Total | Por el **régimen** | Por la **pérdida SSL** | p (SSL vs control) |
|---|---|---|---|---|
| UNM r10 | +.0487 | +.0395 (81 %) | +.0092 (19 %) | 0.245 |
| UNM all-lateral | +.0587 | +.0598 (102 %) | **−.0011 (−2 %)** | 0.792 |
| INCA patient10 | +.0213 | +.0176 (83 %) | +.0036 (17 %) | 0.405 |

**El mejor resultado SSL de la tesis (MT all-lateral, .860) se reproduce entero
con `lambda_u = 0` (.861).** En los tres escenarios, la contribución de la
pérdida de consistencia no se distingue del ruido entre semillas.

Consecuencia para la redacción: lo medido sigue siendo cierto —el pipeline con
datos no etiquetados rinde más que el supervisado— pero **la atribución al
término de consistencia no se sostiene**. La afirmación defendible es que
mejora el *régimen* semi-supervisado, no la pérdida.

### 10b. El pool sigue importando con la pérdida apagada

Con `lambda_u = 0` en ambos lados, pasar de r10 (3.937 frames) a all-lateral
(74.774) sube el F1:

- sin parear (5 vs 3): **+.0203**, Welch p = 0.033, separación 15/15 pares
- pareado en las semillas 0,1,2: **+.0171** (sd .0151), p = 0.188
  - deltas por semilla: +.0345 / +.0095 / +.0074

La dirección es consistente, pero la magnitud está dominada por la semilla 0
(`control_lambda0_r10/seed_0` = .8242, la más baja de las cinco) y **el
contraste pareado no es significativo**. Con 3 semillas y sd .0151, la potencia
para detectar un efecto de ese tamaño es del **7 %**; harían falta 9 semillas
para llegar al 80 %.

**Mecanismo propuesto (pendiente de comprobación).** Con `lambda_u = 0` los no
etiquetados no aportan gradiente, así que sólo queda una vía: el forward del
estudiante sobre ellos se ejecuta igual (`src/train.py`, `logits_u = model(xs_u)`)
en modo `train()`, y actualiza las estadísticas de las **100 capas de BatchNorm**
de U-Net++/efficientnet-b3. Un pool distinto ⇒ estadísticas distintas ⇒ modelo
distinto al evaluar.

Descartado que sea "entrenó más": el bucle recorre el cargador **etiquetado**
(`for it, batch in enumerate(loader)`) y el no etiquetado se recicla con
`next()`, de modo que el número de pasos no depende del tamaño del pool.

**Experimento en curso.** Flag `freeze_bn_on_unlabeled` (defaults, `False` por
defecto): pone `momentum = 0` en las capas BatchNorm durante ese forward, lo que
congela los buffers dejando la salida del forward idéntica (verificado:
diferencia máxima 0.0; `eval()` no serviría porque cambiaría la normalización).
Bloques A/B/C del notebook 16, 5 semillas, salida en
`runs_control_lambda0_bnfrozen/`. Cierra el cuadro:

| | BN se actualiza | BN congelado |
|---|---|---|
| **λ = 0** | .8410 / .8613 | pendiente |
| **λ = 0.05** | .8502 / .8602 | pendiente |

---

## Clasificación (main vs pilot)

- **Main (multi-seed):** UNM pool study (PL/MT × política × radios), UNM arquitecturas, UNM label-efficiency, INCA SSL (full + patient fractions).
- **Referencia contextual single-run:** nnU-Net supervisado (todas las fracciones).
- **Pilot / robustez limitada:** SSL en otros modelos (MT r15 U-Net/FPN); U-Net++ from-scratch.
- **Single-run (marcar explícito):** MT DeepLabV3+ r15; nnU-Net self-training; nnU-Net MT online 150 ep.

## Estado de la auditoría vs tesis_final_v28.tex

Todos los números de resultados de la tesis fueron verificados contra estos archivos y **coinciden** (arquitecturas, pool study, label-efficiency, INCA 10/25/50%, dataset table 968/218/634, INCA unlabeled 31,260). No se encontraron errores numéricos. NO están en la tesis (existen en archivos): from-scratch, nnU-Net SSL (self-training + MT online), pilotos SSL-otros-modelos.

⚠️ **Pendiente de reflejar en la tesis: la sección 10.** La verificación de arriba
compara números, no atribuciones. Los controles `lambda_u = 0` muestran que la
mejora atribuida al SSL viene del régimen y no de la pérdida de consistencia, lo
que afecta a cómo se enuncian las contribuciones 1 y 2 aunque ningún número de
las tablas cambie.
