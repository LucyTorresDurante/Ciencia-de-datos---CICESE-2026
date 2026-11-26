# Guía rápida · `02_dataset_gold`

`datos/gold_1400.csv`, `datos/estratos.json` y las 1,400 preetiquetas binarias (`datos/preetiquetas.jsonl`) ya están generados. Este es el orden para completar el etiquetado humano y el subconjunto de los 21 síntomas del BDI-II.

---

## Paso 1 · Revisión binaria de los 1,400 posts (`revisar.py`)

No hace falta correr el notebook antes; puedes empezar directo desde una terminal:

**En PowerShell (Windows):**
```powershell
cd "d:\jupyterlabcicese\lucy\Final final ora si el chilo\02_dataset_gold"
py revisar.py
```

**En WSL:**
```bash
cd "/mnt/d/jupyterlabcicese/lucy/Final final ora si el chilo/02_dataset_gold"
python3 revisar.py
```

**Controles:**
- `1`: marcar como **positivo** (expresa depresión → `gold = 1`)
- `2`: marcar como **negativo** (no expresa depresión → `gold = 0`)
- `9`: marcar como dudoso (queda fuera de métricas)
- `s`: saltar por ahora
- `u`: deshacer la última acción
- `q`: salir (cada pulsación ya guarda al instante en `datos/gold_1400.csv` y `datos/revision.jsonl`; al volver a abrir continúas exactamente donde te quedaste)

---

## Paso 2 · Correr `02_construccion_dataset_gold.ipynb`

Cuando termines el Paso 1, abre `02_construccion_dataset_gold.ipynb` en JupyterLab y córrelo de arriba a abajo:

1. **Secciones 1 y 2:** como `SOBRESCRIBIR_GOLD = False` y los 1,400 IDs ya están en `datos/preetiquetas.jsonl`, **no** vuelve a muestrear ni vuelve a gastar llamadas en Mistral, y respeta tus etiquetas humanas (`gold` y `nota`).
2. **Sección 3:** genera `../03_adaptacion_mlm/artefactos/mlm_corpus.parquet` excluyendo los 1,400 IDs y verifica 0 fuga (~15 s).
3. **Sección 4:** calcula la tabla de concordancia (**Humano vs. GLM vs. StartLux**) y la prevalencia reponderada a población con `datos/estratos.json`.
4. **Sección 5:** toma todos tus posts confirmados como `gold == 1` y preetiqueta los 21 síntomas del BDI-II para generar `datos/gold_bdi21.csv`. Viene separada en **dos celdas independientes** (ejecuta solo la que prefieras; ambas guardan en la misma caché `datos/preetiquetas_bdi21.jsonl`):
   - **Opción A · GLM (`zai-glm-5-3`):** requiere tu clave en variable de entorno (**no** la pegues en el notebook):
     ```bash
     export MISTRAL_API_KEY="tu_clave_aqui"
     ```
     *(o en PowerShell antes de abrir Jupyter: `$env:MISTRAL_API_KEY="tu_clave_aqui"`)*
   - **Opción B · StartLux local (`/v1/systemone`):** consulta tu servidor local en `127.0.0.1:8889` con el protocolo `state.post` + preguntas `type: "noul"` (sin API externa).
   - Después corre la celda siguiente para construir `datos/gold_bdi21.csv`.

---

## Paso 3 · Revisión de los 21 síntomas BDI-II (`revisar_sintomas.py`)

Cuando la Sección 5 termine de crear `datos/gold_bdi21.csv`, abre la terminal en `02_dataset_gold/`:

**En PowerShell (Windows):**
```powershell
cd "d:\jupyterlabcicese\lucy\Final final ora si el chilo\02_dataset_gold"
py revisar_sintomas.py
```

**En WSL:**
```bash
cd "/mnt/d/jupyterlabcicese/lucy/Final final ora si el chilo/02_dataset_gold"
python3 revisar_sintomas.py
```

**Controles:**
- `Enter` o `a`: aceptar los síntomas marcados con `[*]` y avanzar
- `1 7 14` (números separados por espacio o coma): activar/desactivar esos síntomas y redibujar antes de guardar
- `+14` / `-7`: agregar o quitar síntomas específicos
- `0`: limpiar todos los síntomas marcados
- `9`: marcar el post como dudoso
- `s`: saltar por ahora
- `u`: deshacer la última acción (restaura el estado previo exacto)
- `q`: guardar progreso y salir para continuar después

---

## Paso 4 · Tabla resumen final

Al terminar `revisar_sintomas.py`, vuelve a correr la última celda de `02_construccion_dataset_gold.ipynb` para imprimir la frecuencia y porcentaje de cada uno de los 21 síntomas del BDI-II confirmados manualmente.
