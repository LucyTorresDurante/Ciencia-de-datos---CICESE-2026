# Guía didáctica · `04_evaluacion_downstream`

Este módulo es la prueba final del proyecto: aquí medimos si adaptar `citiusLTL/DisorBERT` al discurso de comunidades trans en Reddit (`DysphorBERT`) realmente mejora la detección de depresión sobre texto trans **sin** echar a perder lo que `DisorBERT` ya sabía hacer en salud mental general.

Todo vive en el cuaderno [`04_evaluacion_bdi21_y_trisk.ipynb`](04_evaluacion_bdi21_y_trisk.ipynb) y los archivos generados se guardan en [`resultados/`](resultados/).

---

## 1. Qué responde cada experimento

### Experimento A · Control de olvido clínico (eRisk 2024, 21 síntomas BDI-II)
Cuando adaptas un modelo de lenguaje con MLM sobre 1.44 millones de publicaciones nuevas, existe el riesgo de *olvido catastrófico*: que aprenda la jerga trans pero pierda sensibilidad hacia los síntomas clínicos del BDI-II.

Para comprobarlo, tomamos `eRisk_2024_dataset_majority.csv` y entrenamos una cabeza multi-etiqueta de 21 salidas (`ClasificadorBDI21`) encima de cada modelo:
* **Agrupación por texto y máscara:** en eRisk 2024 una misma oración puede aparecer anotada solo para algunos síntomas. Por eso cada texto único tiene un vector `labels` de 21 bits y una `mask` que indica en cuáles síntomas sí existe etiqueta; la pérdida (`BCEWithLogitsLoss`) solo se calcula donde `mask == 1`.
* **Partición 70 / 15 / 15 por usuario:** dividimos usando `user_prefix = docid.split("_")[0]` para que ningún usuario ni ningún texto se repita entre entrenamiento, validación y prueba.

### Experimento B1 · Transferencia cero-shot al conjunto gold trans (`T-Risk`)
Aquí ponemos a prueba una pregunta muy limpia: si tomamos los detectores de los 21 síntomas entrenados **únicamente** con eRisk 2024 (sin haber visto ni una sola etiqueta humana de `gold_1400.csv`), ¿cuál distingue mejor las publicaciones con depresión en comunidades trans?

* Para cada publicación de `gold_1400.csv`, el modelo predice las 21 probabilidades `p01..p21` y tomamos como puntuación binaria de riesgo `max(p01..p21)`: si al menos un síntoma clínico del BDI-II supera el umbral, el texto se marca como de riesgo.
* Si además ya revisaste síntomas en `../02_dataset_gold/datos/gold_bdi21.csv` (`revisado == "1"`), el sub-experimento **B1-síntomas** compara directamente `p01..p21` contra tus columnas humanas `s01..s21`.

### Experimento B2 · Validación cruzada 5-fold en `gold_1400.csv`
En el experimento B2 sí entrenamos un clasificador binario directamente sobre tus etiquetas humanas `0/1` de `gold_1400.csv`, usando `StratifiedGroupKFold` de 5 particiones agrupado por `author_hash`. Agrupar por autor evita que el modelo reconozca el estilo de una misma persona entre entrenamiento y validación.

---

## 2. Por qué reportamos métricas en muestra y reponderadas

`gold_1400.csv` no es una muestra aleatoria simple: se construyó estratificando por subreddit y por banda de probabilidad (`0.00-0.10`, `0.10-0.45`, etc.) para tener suficientes casos difíciles y positivos.

Por eso el cuaderno calcula cada métrica de dos maneras:
* **En muestra:** el desempeño directo sobre las publicaciones etiquetadas de `gold_1400.csv`.
* **Reponderada a población (`estratos.json`):** cada observación recibe un peso proporcional al tamaño real de su estrato en Reddit, de modo que el `PR-AUC`, `ROC-AUC`, `F1`, precisión y *recall* reflejen mejor cómo se comportaría el modelo en todo el corpus.

Además, sobre las métricas reponderadas se calcula un intervalo de confianza al 95% mediante *bootstrap* pareado (`N_BOOT = 1000`) y el valor *p* pareado frente a `DisorBERT_base`.

---

## 3. Descubrimiento automático de modelos y caché

No hace falta editar el código cada vez que termines un nuevo entrenamiento MLM en `03_adaptacion_mlm`. Al iniciar, el cuaderno siempre incluye `DisorBERT_base` (`citiusLTL/DisorBERT`) y busca automáticamente cualquier carpeta `../03_adaptacion_mlm/artefactos/disorbert_trans_*` que contenga `modelo_final/model.safetensors`.

Si corres después la ablación sin léxicos (`disorbert_trans_ninguno`) o la variante con `PROB_LEX_DEP = 0.5` (`disorbert_trans_depresion+trans_pdep50_autor`), aparecerán solas en la comparación.

Para no repetir trabajo en GPU:
* `REENTRENAR = False` reutiliza los checkpoints y predicciones guardadas dentro de `resultados/<modelo>/` (`erisk_bdi21_best.pt`, `erisk_test_preds.npz`, `gold1400_bdi21_probs.parquet` y `cv_oof_preds.csv` fold por fold).
* Si `gold_1400.csv` aún no tiene suficientes etiquetas humanas (`< 20` para B1 o `< 50` / `< 10` por clase para B2), o si aún no existe `gold_bdi21.csv`, el cuaderno muestra un aviso claro y continúa sin fallar: así puedes ejecutar desde hoy el Experimento A y volver a correr las secciones B1 y B2 conforme avances con el etiquetado.

---

## 4. Cómo ejecutarlo y qué archivos genera

Abre JupyterLab en el entorno WSL `vllm`, entra a `04_evaluacion_downstream/` y ejecuta `04_evaluacion_bdi21_y_trisk.ipynb` de arriba a abajo. Al terminar encontrarás en `resultados/`:

* Subcarpetas por modelo con checkpoints, predicciones y métricas en JSON/CSV/Parquet.
* Gráficas listas para insertar (`fig_erisk_por_sintoma.png` y `fig_trisk_comparacion.png` a 200 dpi).
* Tablas en formato LaTeX `booktabs` (`tabla_erisk_bdi21.tex`, `tabla_trisk_binario.tex` y `tabla_trisk_bdi21.tex` según los experimentos disponibles) junto con casos cualitativos donde el vocabulario de `lexico_trans.txt` marca la diferencia frente a `DisorBERT_base`.
