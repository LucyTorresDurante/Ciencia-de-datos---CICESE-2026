# DysphorBERT — Detección de depresión y síntomas BDI-II en comunidades trans en Reddit

Proyecto final de posgrado (Inducción a la Ciencia de Datos, CICESE) sobre la tercera adaptación de dominio de [`citiusLTL/DisorBERT`](https://huggingface.co/citiusLTL/DisorBERT) al discurso en línea de comunidades trans en Reddit y su evaluación *downstream* en población general (`eRisk 2024`) y en el conjunto *gold* estratificado `T-Risk`.

## Cuaderno principal y modelo publicado
- **[`DysphorBERT_proyecto_final.ipynb`](DysphorBERT_proyecto_final.ipynb):** cuaderno principal ejecutado de principio a fin con contadores consecutivos y todas sus salidas visibles (EDA, verificación anti-fuga, comparación de modelos MLM y *downstream*, *bootstrap* pareado, análisis cualitativo de errores, conclusiones y declaración de uso de IA generativa). Carga los artefactos generados por los cuadernos modulares de extracción y GPU incluidos en `01`–`04`.
- **Modelo en Hugging Face:** [`LucyTD/DysphorBERT`](https://huggingface.co/LucyTD/DysphorBERT) (pesos de `DysphorBERT v1`, licencia **CC BY-NC 4.0**).

## Estructura del repositorio
- `DysphorBERT_proyecto_final.ipynb`: cuaderno principal consolidado.
- `01_datos_reddit/`: extracción desde volcados mensuales de Reddit (`.zst`), filtrado y pre-clasificación (`bdi_prob`) consolidada en `bdi_total.parquet`.
- `02_dataset_gold/`: muestreo estratificado de 1,400 publicaciones (`datos/gold_1400.csv`), `datos/estratos.json`, anotación de 21 síntomas BDI-II (`datos/gold_bdi21.csv`), herramientas interactivas de consola (`revisar.py`, `revisar_sintomas.py`) y guía (`README.md`).
- `03_adaptacion_mlm/`: adaptación MLM en GPU (`DysphorBERT v1`, `DysphorBERT v2` y ablación `MLM-ninguno`), léxico trans por *keyness* (`artefactos/lexico_trans.txt`), curvas (`artefactos/curvas_mlm.png`) y guía didáctica (`README.md`).
- `04_evaluacion_downstream/`: evaluación en `eRisk 2024` (Experimento A) y `T-Risk` (Experimentos B1 cero-shot y B2 5-*fold* CV agrupada por autor), tablas y figuras en `resultados/`.

## Origen de los datos y reproducibilidad
Por **privacidad** (relatos sensibles de usuarios en comunidades de salud mental/trans), **acuerdos de uso** y **tamaño ($> 50$ MB)**, **no se redistribuye ningún texto de Reddit ni de `eRisk 2024`** en este repositorio (los volcados `.zst`, archivos `.parquet`, CSVs con publicaciones, checkpoints y pesos `.safetensors` están excluidos en `.gitignore`). El repositorio incluye únicamente código, cuadernos ejecutados, métricas agregadas, léxicos, tamaños de estrato (`estratos.json`) y figuras:

1. **Publicaciones de Reddit (2005–2025-12, 247 meses):** obtenidas de los *torrents* mensuales del proyecto [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) (`RS_YYYY-MM.zst`), filtradas para siete subreddits trans (`asktransgender`, `ftm`, `MtF`, `trans`, `NonBinary`, `TransLater`, `TransMasc`) con `01_datos_reddit/01_extraccion_y_filtrado_reddit.ipynb`. No se redistribuyen por privacidad de los usuarios y por tamaño ($\approx 1$ TB en `.zst` y $> 850$ MB en `.parquet`).
2. **Conjunto *gold* `T-Risk` (elaboración propia):** muestra estratificada de 1,400 publicaciones (`gold_1400.csv`, 200 por subreddit en 5 bandas de `bdi_prob`) anotada manualmente en forma binaria (180 positivos, 1,220 negativos) y en los 21 síntomas del BDI-II sobre los 180 positivos (`gold_bdi21.csv`), construida con `02_dataset_gold/`. Por privacidad de las publicaciones originales de Reddit, no se redistribuyen los textos; en el repositorio se conservan los tamaños poblacionales de los 35 estratos (`02_dataset_gold/datos/estratos.json`) y las herramientas de muestreo/revisión.
3. **eRisk 2024 (21 síntomas BDI-II):** conjunto `eRisk_2024_dataset_majority.csv` perteneciente al taller [CLEF eRisk 2024](https://erisk.irlab.org/), facilitado con fines de investigación académica por nuestra compañera Cielo (posgrado CICESE). Para reproducirlo, debe solicitarse directamente a los organizadores de eRisk bajo su acuerdo de uso; no se redistribuye en este repositorio.
4. **Modelo base:** [`citiusLTL/DisorBERT`](https://huggingface.co/citiusLTL/DisorBERT) (licencia **CC BY-NC 4.0**; cualquier derivado hereda la restricción de uso no comercial).
5. **Pre-clasificador local:** [`startlux-models/StartLux-Decision-v1.1-4B-AutoThink`](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink) servido localmente con `vLLM`.

## Requisitos
- Python 3.12+
- PyTorch 2.13+ (ROCm o CUDA)
- `transformers >= 5.0`, `datasets`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`, `tqdm`
- Para el pre-etiquetado opcional con la API de Mistral en `02_dataset_gold`, definir la variable de entorno `MISTRAL_API_KEY` (nunca guardar claves dentro de los cuadernos).
