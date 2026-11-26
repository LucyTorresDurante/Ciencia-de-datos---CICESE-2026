# Guía didáctica · `03_adaptacion_mlm`

Este módulo hace la **tercera adaptación de dominio** de tu proyecto: toma [`citiusLTL/DisorBERT`](https://huggingface.co/citiusLTL/DisorBERT) (que ya venía adaptado de texto general a salud mental) y lo adapta al vocabulario y formas de expresarse de las comunidades trans en Reddit. Al modelo adaptado lo llamamos **DysphorBERT**.

Aquí **todavía no se clasifica depresión**: primero se enseña al modelo a "entender el idioma" de estos subreddits sin usar etiquetas manuales.

---

## 1. Conceptos clave (qué hace cada pieza del notebook)

### ¿Qué es MLM (*Masked Language Modeling*)?
Es la tarea de "rellenar el espacio en blanco" con la que se entrenan los modelos tipo BERT. Se toman textos reales, se tapan algunas palabras con `[MASK]` y el modelo intenta adivinar qué palabra iba ahí leyendo el contexto alrededor.
* **Ventaja:** aprovecha tus **1.44 millones de publicaciones** sin necesidad de etiquetarlas a mano.
* **Por qué no usar solo textos con depresión:** si entrenaras MLM solo con posts depresivos, el modelo vería menos ejemplos de jerga cotidiana trans (*HRT*, *t4t*, *boymode*, *stealth*). Usar todo el corpus le da un vocabulario más rico.

### ¿Por qué el tokenizador parte algunas palabras? (WordPiece)
BERT tiene un diccionario fijo de **30,522 piezas** (*tokens*). Las palabras comunes están enteras, pero la jerga específica suele partirse en subpalabras con `##`, por ejemplo:
* `dysphoria` → `['d', '##ys', '##ph', '##oria']`
* `deadname` → `['dead', '##name']`

Si usáramos enmascarado normal pieza por pieza y solo se tapara `##oria` dejando `d ##ys ##ph`, adivinar sería trivial por pura ortografía y el modelo no aprendería el significado real de la palabra en la frase.

### ¿Qué es un *Collator* y por qué sí es necesario aquí?
En PyTorch / Hugging Face, mientras el modelo entrena por lotes (*batches*), el **collator** es la función que toma un puñado de ejemplos crudos (por ejemplo 32 bloques de texto) y los prepara justo antes de pasarlos a la GPU.

En este proyecto **no basta el collator estándar** de Hugging Face porque queremos usar **`CollatorGuiado`**, que hace tres cosas al vuelo:
1. **Enmascarado por palabra completa:** si una palabra pertenece a tus léxicos, tapa **todas** sus piezas juntas (por ejemplo los 4 tokens de `dysphoria` a la vez), obligando al modelo a usar el contexto de la oración.
2. **Prioridad por léxico:** según `LEXICO_MODO`:
   * `"depresion+trans"` (default actual): tapa palabras de depresión con probabilidad `PROB_LEX_DEP = 0.5` (la corrida original usó `1.0`), palabras trans con `PROB_LEX_TRANS = 0.7`, y el resto de tokens al azar al `20%`. Con `0.5`, el modelo sigue viendo los términos de depresión como contexto intacto la mitad de las veces.
   * `"depresion"`: solo guía las palabras de depresión y enmascara el resto al `20%`.
   * `"ninguno"` (la ablación de control): ignora los léxicos y tapa al azar el `20%` de los tokens.
3. **Regla 80 / 10 / 10 y tope del 40%:** sobre los tokens elegidos para predecir, el 80% se reemplaza por `[MASK]`, el 10% por una palabra aleatoria y el 10% se deja igual (para que el modelo no dependa de ver siempre `[MASK]`), recortando si un bloque pasa de `MAX_MASCARA = 0.40`.

### Empaquetado en bloques de 256 tokens y lote efectivo
* **Empaquetado (`MAX_LEN = 256`):** en lugar de rellenar posts cortos con ceros vacíos (*padding*, que desperdician GPU), todos los posts tokenizados se concatenan y se cortan en bloques exactos de 256 tokens.
* **Lote efectivo (`BATCH = 32`, `GRAD_ACCUM = 2`):** la GPU procesa 32 bloques por paso y acumula los gradientes de 2 pasos seguidos antes de actualizar los pesos. Así se comporta matemáticamente como un lote de **64 bloques** gastando la mitad de VRAM.
* **`bf16` y `LR = 3e-5`:** `bf16` es precisión mixta nativa y estable en tu Radeon RX 9070 XT; la tasa de aprendizaje `3e-5` con 6% de *warmup* (subida gradual inicial) evita destruir lo que DisorBERT ya sabía.

### ¿Qué es la perplejidad (PPL) y cuál es comparable?
La perplejidad es $\text{PPL} = e^{\text{loss}}$: mide entre cuántas opciones duda el modelo al adivinar los tokens tapados. **Mientras más baja, mejor.**
* **PPL guiada** (`77.505 → 5.797` en `depresion+trans`): mide sobre las mismas máscaras difíciles del entrenamiento (100% depresión + 70% trans). Sirve para comparar ese modelo contra sí mismo antes y después, pero **no** contra otro modo que usa máscaras distintas.
* **PPL neutral al 15%** (`35.655 → 4.600`): usa un `colador_neutral` idéntico y con semilla fija (`SEED + 7`) para cualquier modo. **Esta sí es la cifra justa** para comparar `depresion+trans` contra `ninguno`.

### Las 16 sondas (*cloze*)
Evalúan qué probabilidad (`logp`) y qué lugar en el ranking (`rango`) asigna el modelo a una palabra concreta en una frase de prueba:
* **10 sondas "en léxico"** (*testosterone, dysphoria, deadname, t4t, stealth...*): muestran qué tanto memorizó/ajustó los términos que el entrenamiento tapó explícitamente.
* **6 sondas "fuera del léxico" (*held-out*)** (*genderqueer, agender, progesterone, transmisogyny, mastectomy, crossdresser*): nunca se guiaron durante el entrenamiento, así que miden **generalización real**. En `depresion+trans` la media fue `+0.644` (mejoró en 4 de 6 y empeoró en `crossdresser` y `transmisogyny`).

### ¿Por qué el `train_loss` parecía el doble que `eval_loss`?
Con `GRAD_ACCUM = 2` en `transformers 5.20.2`, el historial guarda la pérdida sumada de los 2 micro-lotes antes de promediar. Por eso en el log original aparecía `3.665` frente a `eval_loss = 1.757`. Al dividir entre `GRAD_ACCUM` en la Sección 7 queda `1.8326`, totalmente coherente con `1.7574`.

### ¿Qué es la evaluación *downstream* (Notebook 04)?
Adaptar con MLM solo mejora la "comprensión del idioma". Para responder tu pregunta de investigación necesitas entrenar **una tarea supervisada encima** (*downstream*, río abajo):
1. Tomas `DisorBERT` base, tu modelo `MLM-ninguno` y tu `DysphorBERT` (`depresion+trans`).
2. A los tres les conectas la misma capa clasificadora nueva y los entrenas con **exactamente las mismas reglas** sobre **eRisk 2024** (21 síntomas BDI-II) y sobre tu **gold trans** (`gold_1400.csv` y `gold_bdi21.csv`).
3. Ahí ves con números reales (F1, recall, PR-AUC) si la adaptación de dominio ayudó a detectar mejor los síntomas en textos trans sin perder desempeño en población general.

---

## 2. Qué se puede mejorar en tu modelo (y estado en el notebook)

Tu modelo ya entrenado (`artefactos/disorbert_trans_depresion+trans`, con `PROB_LEX_DEP = 1.0` y split por publicación) está bien hecho: usó 1.44M posts limpios, excluyó por ID, por hash y por los 1,347 autores del gold, y convergió sin inestabilidades. Esa carpeta **se conserva intacta** y la Sección 7 la compara automáticamente contra cualquier experimento nuevo.

Las tres mejoras técnicas discutidas para el paper quedan así:

1. **Suavizar `PROB_LEX_DEP` de `1.0` a `0.5` (ya es el default del notebook):** al poner `1.0` en la corrida original, el modelo nunca vio sin tapar ninguna de las 126 palabras de depresión; con `0.5` las ve como contexto intacto la mitad de las veces. Queda listo como experimento pendiente de entrenar.
2. **Split de validación agrupado por autor (`SPLIT_POR_AUTOR = True`, ya es el default del notebook):** separa todos los posts de cada autor en el mismo lado del split (tratando `"[deleted]"` y autores genéricos como grupos individuales `anon_<id>`) para evitar cualquier optimismo en la PPL de validación. También queda listo en este experimento pendiente de entrenar.
3. **Mezcla contra olvido catastrófico:** si en el Notebook 04 vieras que `DysphorBERT` cayó frente a `DisorBERT` en eRisk 2024, se mitiga bajando `LR` a `1e-5` o mezclando ~10–15% de texto general/clínico durante el MLM.

---

## 3. Esquema de carpetas, cómo correr, pausar y reanudar

Para no mezclar ni sobrescribir experimentos, `SALIDA` se construye automáticamente con `ETIQUETA_CORRIDA`:
* **Default mejorado** (`LEXICO_MODO = "depresion+trans"`, `PROB_LEX_DEP = 0.5`, `SPLIT_POR_AUTOR = True`): guarda en `artefactos/disorbert_trans_depresion+trans_pdep50_autor/`.
* **Ablación de control** (`LEXICO_MODO = "ninguno"`, `SPLIT_POR_AUTOR = True`): guarda en `artefactos/disorbert_trans_ninguno_autor/`.
* **Corrida original** (ya entrenada, intacta): vive en `artefactos/disorbert_trans_depresion+trans/`.

Pasos para correr o reanudar:
1. Abre `03_adaptacion_dominio_dysphorbert.ipynb` en JupyterLab usando el kernel del entorno `vllm` (ROCm).
2. Con los defaults actuales (`LEXICO_MODO = "depresion+trans"`, `PROB_LEX_DEP = 0.5`, `SPLIT_POR_AUTOR = True`, `ENTRENAR_SI_EXISTE_MODELO_FINAL = False`) entrenas la versión mejorada en `artefactos/disorbert_trans_depresion+trans_pdep50_autor/`.
3. Para repetir la ablación sin léxicos, cambia `LEXICO_MODO = "ninguno"` (escribirá en `artefactos/disorbert_trans_ninguno_autor/`).
4. Para **pausar**: detén el kernel en cualquier momento; los checkpoints se guardan cada `1,000` pasos en `SALIDA/checkpoints/`. Al volver a correr las celdas desde arriba, `huella.json` verifica que no cambiaste hiperparámetros y retoma desde el último checkpoint.
5. Al terminar (o en cualquier punto si solo corres `imports` y `config`), corre la **Sección 7**: descubre automáticamente todas las carpetas `artefactos/disorbert_trans_*` con `trainer_state.json` (incluida la corrida original), actualiza la tabla comparativa y guarda `artefactos/curvas_mlm.png`.

> **Licencia:** `citiusLTL/DisorBERT` está publicado bajo **CC BY-NC 4.0**, por lo que `DysphorBERT` hereda la restricción de uso no comercial.
