#!/usr/bin/env python3
"""Revisión del preetiquetado con cuota estratificada por subreddit.

Versión consolidada sobre un único archivo ``datos/gold_1400.csv`` y bitácora
``datos/revision.jsonl``.

Objetivo por defecto: CUOTA_POR_SUBREDDIT etiquetas humanas (0/1) por subreddit,
repartidas por banda de bdi_prob en las mismas proporciones que la muestra.

    1 = positivo (depresión)              (gold = 1)
    2 = negativo (no depresión)           (gold = 0)
    9 = dudoso                            (gold = 9, fuera de métricas)
    s = saltar        u = deshacer        q = guardar y salir

Uso:
    cd "Final final ora si el chilo/02_dataset_gold"
    python3 revisar.py      (WSL)      ·      py revisar.py   (Windows)
"""
import csv
import json
import os
import shutil
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
DATOS_DIR = CARPETA / "datos"
ARCHIVO_GOLD = DATOS_DIR / "gold_1400.csv"
LOG = DATOS_DIR / "revision.jsonl"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── qué y cuánto ──────────────────────────────────────────────────────────
CUOTA_POR_SUBREDDIT = 200        # etiquetas humanas (0/1) por subreddit
SIN_CUOTA = False                # True = recorrer todo sin límite

FRACCIONES_BANDA = {             # proporción de la cuota por banda
    "0.00-0.10": 0.30,
    "0.10-0.45": 0.20,
    "0.45-0.65": 0.30,
    "0.65-0.90": 0.10,
    "0.90-1.00": 0.10,
}
# orden de presentación: primero donde los modelos discrepan
BANDAS_PRIORITARIAS = ["0.45-0.65", "0.65-0.90", "0.00-0.10", "0.10-0.45", "0.90-1.00"]

MAX_TEXTO = 3000
ANSI = True
SALTAR_COMPROBACION = False

TABLA = {"columnas": [], "filas": []}
HECHO = defaultdict(int)         # (subreddit, banda) -> gold 0/1 ya puestos
TOTAL_SUB = defaultdict(int)


# ── terminal ──────────────────────────────────────────────────────────────
def habilitar_ansi():
    if os.name == "nt":
        os.system("")


def limpiar():
    if not ANSI:
        print("\n" + "=" * 72)
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def leer_tecla():
    if os.name == "nt":
        import msvcrt
        while True:
            t = msvcrt.getwch()
            if t in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            if t.isprintable():
                return t.lower()
    import termios
    import tty
    fd = sys.stdin.fileno()
    viejo = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            t = sys.stdin.read(1)
            if t.isprintable():
                return t.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, viejo)


# ── cuota: solo cuenta gold 0/1, nunca 9 ni vacío ─────────────────────────
def es_etiqueta(valor):
    """True si el valor cuenta para la cuota (gold 0 o 1 ya puesto)."""
    return str(valor).strip() in ("0", "1")


def ajustar_hecho(subreddit, banda, delta):
    """Suma delta a HECHO y evita cuotas negativas por dobles descuentos."""
    clave = (subreddit, banda)
    HECHO[clave] = max(0, HECHO[clave] + delta)


# ── datos ─────────────────────────────────────────────────────────────────
def cargar():
    HECHO.clear()
    TOTAL_SUB.clear()
    if not ARCHIVO_GOLD.exists():
        raise FileNotFoundError(f"No encuentro el dataset consolidado: {ARCHIVO_GOLD}")
    with open(ARCHIVO_GOLD, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        TABLA["columnas"] = list(lector.fieldnames or [])
        TABLA["filas"] = list(lector)
    for fila in TABLA["filas"]:
        sub = fila.get("subreddit", "")
        TOTAL_SUB[sub] += 1
        if es_etiqueta(fila.get("gold") or ""):
            HECHO[(sub, fila.get("banda", ""))] += 1


def item_de(i, fila):
    return {"idx": i,
            "subreddit": fila.get("subreddit", ""),
            "orden": fila.get("orden", ""),
            "anio": fila.get("anio", ""),
            "palabras": fila.get("palabras", ""),
            "bdi_prob": fila.get("bdi_prob", ""),
            "banda": fila.get("banda", ""),
            "titulo": fila.get("titulo", ""),
            "texto": fila.get("texto", ""),
            "api": int(float(fila.get("api"))),
            "motivo": (fila.get("api_motivo") or "").strip()}


def construir_cola():
    """Cuota por subreddit, repartida por banda, y balanceada entre subreddits."""
    pend = defaultdict(list)
    for i, fila in enumerate(TABLA["filas"]):
        if (fila.get("gold") or "").strip():
            continue
        if (fila.get("api") or "").strip() == "":
            continue
        pend[fila.get("subreddit", "")].append(item_de(i, fila))

    colas_por_sub = {}
    for sub, items in sorted(pend.items()):
        items.sort(key=lambda x: int(float(x["orden"] or 0)))
        if SIN_CUOTA:
            colas_por_sub[sub] = items
            continue

        por_banda = defaultdict(list)
        for it in items:
            por_banda[it["banda"]].append(it)

        elegidos = []
        for banda in BANDAS_PRIORITARIAS:
            objetivo = int(round(CUOTA_POR_SUBREDDIT * FRACCIONES_BANDA.get(banda, 0)))
            falta = max(0, objetivo - HECHO[(sub, banda)])
            elegidos.extend(por_banda[banda][:falta])

        faltan_total = CUOTA_POR_SUBREDDIT - sum(
            HECHO[(sub, b)] for b in FRACCIONES_BANDA)
        if faltan_total > len(elegidos):
            ya = {id(x) for x in elegidos}
            for it in items:
                if len(elegidos) >= faltan_total:
                    break
                if id(it) not in ya:
                    elegidos.append(it)
        colas_por_sub[sub] = elegidos

    cola, i = [], 0
    subs = list(colas_por_sub)
    while any(i < len(colas_por_sub[s]) for s in subs):
        for s in subs:
            if i < len(colas_por_sub[s]):
                cola.append(colas_por_sub[s][i])
        i += 1
    return cola


def guardar_tabla():
    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVO_GOLD.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=TABLA["columnas"])
        escritor.writeheader()
        escritor.writerows(TABLA["filas"])
    os.replace(tmp, ARCHIVO_GOLD)


def anotar_log(fila, accion, valor):
    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": fila.get("id"), "accion": accion, "gold": valor,
                            "api": fila.get("api"), "subreddit": fila.get("subreddit"),
                            "orden": fila.get("orden"), "banda": fila.get("banda"),
                            "fecha": time.strftime("%Y-%m-%d %H:%M:%S")},
                           ensure_ascii=False) + "\n")


def aplicar(item, valor, accion=""):
    fila = TABLA["filas"][item["idx"]]
    anterior = fila.get("gold") or ""
    fila["gold"] = str(valor)
    guardar_tabla()
    anotar_log(fila, accion, valor)

    if es_etiqueta(anterior):
        ajustar_hecho(item["subreddit"], item["banda"], -1)
    if es_etiqueta(valor):
        ajustar_hecho(item["subreddit"], item["banda"], +1)
    return anterior


# ── pantalla ──────────────────────────────────────────────────────────────
def mostrar(item, i, total, stats, t0):
    ancho = min(shutil.get_terminal_size((90, 30)).columns, 100)
    limpiar()
    hecho = sum(HECHO[(item["subreddit"], b)] for b in FRACCIONES_BANDA)
    gold = sum(min(HECHO[(item["subreddit"], b)], round(CUOTA_POR_SUBREDDIT * f))
               for b, f in FRACCIONES_BANDA.items())
    completo = gold >= CUOTA_POR_SUBREDDIT
    nota = " · gold completo, esto ya es entrenamiento" if completo else ""
    print("=" * ancho)
    print(f" {item['subreddit']} #{item['orden']}  ·  {item['anio']}  ·  "
          f"{item['palabras']} palabras  ·  bdi_prob {item['bdi_prob']}")
    print(f" {item['subreddit']}: gold {gold}/{CUOTA_POR_SUBREDDIT} · "
          f"{hecho} etiquetas totales{nota}   ·   banda {item['banda']}")
    linea_ia = f" IA dice: {item['api']}"
    if item["motivo"]:
        linea_ia += f'   ·   "{item["motivo"]}"'
    print(linea_ia)
    print("-" * ancho)
    print(textwrap.fill(" ".join(str(item["titulo"]).split()), width=ancho))
    print()
    texto = str(item["texto"])
    recortado = len(texto) > MAX_TEXTO
    for parrafo in texto[:MAX_TEXTO].split("\n"):
        if not parrafo.strip():
            print()
        else:
            print(textwrap.fill(" ".join(parrafo.split()), width=ancho))
    if recortado:
        print(f"\n  [... recortado a {MAX_TEXTO} de {len(texto)} caracteres]")
    print("-" * ancho)
    print("  [1] positivo (1)    [2] negativo (0)    "
          "[9] dudoso    [s] saltar    [u] deshacer    [q] salir")
    hechos = stats["positivos"] + stats["negativos"] + stats["dudosos"]
    partes = [f"{i}/{total}", f"positivos {stats['positivos']}",
              f"negativos {stats['negativos']}", f"dudosos {stats['dudosos']}",
              f"saltados {stats['saltados']}"]
    if hechos:
        faltan = int((total - i) * (time.time() - t0) / hechos)
        partes.append(f"~{faltan // 60} min restantes")
    print("  " + "  ·  ".join(partes))


# ── principal ─────────────────────────────────────────────────────────────
def main():
    if not SALTAR_COMPROBACION and not sys.stdin.isatty():
        print("Esto necesita una terminal de verdad; dentro de Jupyter no funciona.")
        print("Abre WSL o PowerShell, entra en la carpeta y ejecuta: python3 revisar.py")
        return

    habilitar_ansi()
    cargar()
    cola = construir_cola()
    total = len(cola)
    if total == 0:
        print("No queda nada por revisar con la cuota actual.")
        print("Si quieres seguir etiquetando, sube CUOTA_POR_SUBREDDIT en este archivo.")
        return

    print(f"{total:,} posts por revisar ({CUOTA_POR_SUBREDDIT} por subreddit). Preparando...")
    time.sleep(0.6)

    stats = {"positivos": 0, "negativos": 0, "dudosos": 0, "saltados": 0}
    pila = []
    i = 0
    t0 = time.time()
    try:
        while i < total:
            item = cola[i]
            mostrar(item, i + 1, total, stats, t0)
            tecla = leer_tecla()

            if tecla == "q":
                break
            if tecla in ("1", "2", "9"):
                valor = {"1": 1, "2": 0, "9": 9}[tecla]
                nombre = {"1": "positivo", "2": "negativo", "9": "dudoso"}[tecla]
                pila.append((i, item, aplicar(item, valor, nombre), tecla))
                stats[{"1": "positivos", "2": "negativos", "9": "dudosos"}[tecla]] += 1
                i += 1
            elif tecla == "s":
                stats["saltados"] += 1
                i += 1
            elif tecla == "u" and pila:
                j, previo, anterior, cual = pila.pop()
                fila_prev = TABLA["filas"][previo["idx"]]
                actual = fila_prev.get("gold") or ""

                if es_etiqueta(actual):
                    ajustar_hecho(previo["subreddit"], previo["banda"], -1)

                fila_prev["gold"] = anterior

                if es_etiqueta(anterior):
                    ajustar_hecho(previo["subreddit"], previo["banda"], +1)

                guardar_tabla()
                anotar_log(fila_prev, "deshacer", anterior)
                stats[{"1": "positivos", "2": "negativos", "9": "dudosos"}[cual]] -= 1
                i = j
    except KeyboardInterrupt:
        pass

    print(f"\n\nListo. positivos {stats['positivos']} · negativos {stats['negativos']} · "
          f"dudosos {stats['dudosos']} · saltados {stats['saltados']}")
    for sub in sorted(TOTAL_SUB):
        hecho = sum(HECHO[(sub, b)] for b in FRACCIONES_BANDA)
        print(f"  {sub:<15} {hecho:>4}/{CUOTA_POR_SUBREDDIT}")
    print(f"\nGuardado en {ARCHIVO_GOLD} y en {LOG}")


if __name__ == "__main__":
    main()
