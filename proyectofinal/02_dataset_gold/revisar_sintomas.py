#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revisar_sintomas.py

Herramienta TUI para revisar los 21 sintomas del BDI-II en posts
positivos de Reddit. Compatible con Windows y WSL.
"""

import csv
import json
import os
import re
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
ARCHIVO = CARPETA / "datos" / "gold_bdi21.csv"
LOG = CARPETA / "datos" / "revision_bdi21.jsonl"

MAX_TEXTO = 3000
SALTAR_COMPROBACION = False

BDI_21 = [
    (1, "Sadness", "Tristeza"),
    (2, "Pessimism", "Pesimismo"),
    (3, "Past Failure", "Fracaso pasado"),
    (4, "Loss of Pleasure", "Perdida de placer"),
    (5, "Guilty Feelings", "Sentimientos de culpa"),
    (6, "Punishment Feelings", "Sentimientos de castigo"),
    (7, "Self-Dislike", "Rechazo hacia si mismo"),
    (8, "Self-Criticalness", "Autocritica"),
    (9, "Suicidal Thoughts or Wishes", "Pensamientos suicidas"),
    (10, "Crying", "Llanto"),
    (11, "Agitation", "Agitacion"),
    (12, "Loss of Interest", "Perdida de interes"),
    (13, "Indecisiveness", "Indecision"),
    (14, "Worthlessness", "Inutilidad o falta de valor"),
    (15, "Loss of Energy", "Perdida de energia"),
    (16, "Changes in Sleeping Pattern", "Cambios en el sueno"),
    (17, "Irritability", "Irritabilidad"),
    (18, "Changes in Appetite", "Cambios en el apetito"),
    (19, "Concentration Difficulty", "Dificultad de concentracion"),
    (20, "Tiredness or Fatigue", "Cansancio o fatiga"),
    (21, "Loss of Interest in Sex", "Perdida de interes en el sexo"),
]

CAMPOS_S = ["s{:02d}".format(numero) for numero in range(1, 22)]

CAMPOS_ESPERADOS = [
    "orden",
    "id",
    "subreddit",
    "anio",
    "palabras",
    "bdi_prob",
    "titulo",
    "texto",
    "pre_modelo",
    "pre_sintomas",
    "pre_motivo",
    "gold_sintomas",
    "revisado",
] + CAMPOS_S

CAMPOS_SNAPSHOT = ["gold_sintomas", "revisado"] + CAMPOS_S


def configurar_salida():
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def limpiar_pantalla():
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def obtener_ancho():
    try:
        ancho = shutil.get_terminal_size(fallback=(140, 40)).columns
    except (OSError, ValueError):
        ancho = 140
    return max(60, min(ancho, 200))


def como_texto(valor):
    return "" if valor is None else str(valor)


def cargar():
    if not ARCHIVO.is_file():
        raise FileNotFoundError("No existe el archivo: {}".format(ARCHIVO))

    with open(ARCHIVO, "r", encoding="utf-8-sig", newline="") as manejador:
        lector = csv.DictReader(manejador)
        campos = list(lector.fieldnames or [])
        filas = [dict(fila) for fila in lector]

    if not SALTAR_COMPROBACION:
        faltantes = [campo for campo in CAMPOS_ESPERADOS if campo not in campos]
        if faltantes:
            raise ValueError(
                "Faltan columnas en {}: {}".format(
                    ARCHIVO.name, ", ".join(faltantes)
                )
            )

    return campos, filas


def construir_cola(filas):
    return [
        indice
        for indice, fila in enumerate(filas)
        if str(fila.get("revisado", "") or "").strip() == ""
    ]


def parsear_numeros(texto):
    piezas = [
        pieza
        for pieza in re.split(r"[\s,]+", str(texto or "").strip())
        if pieza
    ]
    if not piezas:
        return None

    numeros = set()
    for pieza in piezas:
        if not pieza.isdigit():
            return None
        numero = int(pieza)
        if numero < 1 or numero > 21:
            return None
        numeros.add(numero)

    return sorted(numeros)


def seleccion_desde(valor):
    return set(parsear_numeros(valor) or [])


def formatear_gold(seleccion):
    return ",".join(str(numero) for numero in sorted(seleccion))


def snapshot(fila):
    return {
        campo: como_texto(fila.get(campo))
        for campo in CAMPOS_SNAPSHOT
    }


def restaurar(fila, estado):
    for campo in CAMPOS_SNAPSHOT:
        fila[campo] = como_texto(estado.get(campo))


def aplicar_guardado(fila, seleccion):
    fila["gold_sintomas"] = formatear_gold(seleccion)
    fila["revisado"] = "1"
    for numero in range(1, 22):
        fila["s{:02d}".format(numero)] = "1" if numero in seleccion else "0"


def aplicar_dudoso(fila):
    fila["gold_sintomas"] = "9"
    fila["revisado"] = "9"
    for campo in CAMPOS_S:
        fila[campo] = ""


def guardar_csv(campos, filas):
    temporal = ARCHIVO.with_name(ARCHIVO.name + ".tmp")

    try:
        with open(temporal, "w", encoding="utf-8-sig", newline="") as manejador:
            escritor = csv.DictWriter(
                manejador,
                fieldnames=campos,
                extrasaction="ignore",
                lineterminator="\n",
            )
            escritor.writeheader()
            for fila in filas:
                escritor.writerow(
                    {campo: fila.get(campo, "") for campo in campos}
                )
            manejador.flush()
            os.fsync(manejador.fileno())

        os.replace(temporal, ARCHIVO)

    except Exception:
        try:
            if temporal.exists():
                temporal.unlink()
        except OSError:
            pass
        raise


def registrar(accion, fila, antes, despues):
    evento = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accion": accion,
        "orden": como_texto(fila.get("orden")),
        "id": como_texto(fila.get("id")),
        "gold_sintomas": despues.get("gold_sintomas", ""),
        "revisado": despues.get("revisado", ""),
        "snapshot_anterior": antes,
        "snapshot_nuevo": despues,
    }

    for campo in CAMPOS_S:
        evento[campo] = despues.get(campo, "")

    with open(LOG, "a", encoding="utf-8", newline="") as manejador:
        manejador.write(
            json.dumps(
                evento,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        manejador.flush()
        os.fsync(manejador.fileno())


def envolver(texto, ancho, limite=None):
    crudo = como_texto(texto)
    crudo = crudo.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)

    recortado = False
    if limite is not None and len(crudo) > limite:
        crudo = crudo[:limite]
        recortado = True

    crudo = crudo.strip()
    if not crudo:
        return ["(vacio)"]

    resultado = []
    for parrafo in crudo.split("\n"):
        if not parrafo.strip():
            resultado.append("")
            continue

        envueltas = textwrap.wrap(
            parrafo,
            width=max(10, ancho),
            break_long_words=True,
            break_on_hyphens=False,
        )
        resultado.extend(envueltas or [""])

    if recortado:
        resultado.append("[... recortado ...]")

    return resultado


def mostrar_bloque(titulo, valor, ancho, limite=None):
    print(titulo + ":")
    for linea in envolver(valor, max(10, ancho - 2), limite):
        print("  " + linea)


def lineas_sintoma(sintoma, seleccion, ancho):
    numero, ingles, espanol = sintoma
    marca = "[*]" if numero in seleccion else "[ ]"
    etiqueta = "{} {:>2}. {} / {}".format(marca, numero, ingles, espanol)

    return textwrap.wrap(
        etiqueta,
        width=ancho,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [etiqueta]


def dibujar(fila, seleccion, numero_post, total_posts, aviso):
    limpiar_pantalla()
    ancho = obtener_ancho()
    separador = "-" * ancho

    print("REVISION BDI-II - POSTS POSITIVOS DE REDDIT")
    print(separador)
    print(
        "Pendiente {}/{} | orden={} | id={}".format(
            numero_post,
            total_posts,
            como_texto(fila.get("orden")),
            como_texto(fila.get("id")),
        )
    )
    print(
        "subreddit={} | anio={} | palabras={} | bdi_prob={}".format(
            como_texto(fila.get("subreddit")),
            como_texto(fila.get("anio")),
            como_texto(fila.get("palabras")),
            como_texto(fila.get("bdi_prob")),
        )
    )
    print(separador)

    mostrar_bloque("TITULO", fila.get("titulo"), ancho)
    mostrar_bloque("PRE-MODELO", fila.get("pre_modelo"), ancho)
    mostrar_bloque("PRE-SINTOMAS", fila.get("pre_sintomas"), ancho)
    mostrar_bloque("PRE-MOTIVO", fila.get("pre_motivo"), ancho)

    if aviso:
        print(">> " + aviso)

    print(separador)
    print("TEXTO (max. {} caracteres):".format(MAX_TEXTO))
    for linea in envolver(fila.get("texto"), max(10, ancho - 2), MAX_TEXTO):
        print("  " + linea)

    print(separador)
    print(
        "SINTOMAS BDI-II (activos: {}; [*]=activo):".format(
            len(seleccion)
        )
    )

    ancho_columna = max(24, (ancho - 3) // 2)

    for indice in range(11):
        izquierda = lineas_sintoma(
            BDI_21[indice], seleccion, ancho_columna
        )
        derecha = (
            lineas_sintoma(BDI_21[indice + 11], seleccion, ancho_columna)
            if indice + 11 < len(BDI_21)
            else []
        )

        for renglon in range(max(len(izquierda), len(derecha))):
            texto_izquierda = (
                izquierda[renglon] if renglon < len(izquierda) else ""
            )
            texto_derecha = (
                derecha[renglon] if renglon < len(derecha) else ""
            )
            print(
                texto_izquierda.ljust(ancho_columna)
                + " "
                + texto_derecha
            )

    print(separador)
    print("Enter/a=guardar y avanzar | 1 7 14=alternar | +14=agregar | -7=quitar")
    print("0=limpiar | 9=dudoso | s=saltar | u=deshacer | q=guardar y salir")
    sys.stdout.flush()


def main():
    configurar_salida()
    if not SALTAR_COMPROBACION and not sys.stdin.isatty():
        print("Esto necesita una terminal de verdad; dentro de Jupyter no funciona.")
        print("Abre WSL o PowerShell, entra en 02_dataset_gold y ejecuta: python3 revisar_sintomas.py")
        return 1

    try:
        campos, filas = cargar()
    except Exception as exc:
        limpiar_pantalla()
        print(
            "ERROR al cargar {}: {}".format(ARCHIVO, exc),
            file=sys.stderr,
        )
        return 1

    cola = construir_cola(filas)

    if not cola:
        limpiar_pantalla()
        print("No hay posts pendientes de revision.")
        return 0

    historial = []
    pos = 0
    aviso = ""

    while pos < len(cola):
        fila = filas[cola[pos]]
        seleccion = seleccion_desde(fila.get("pre_sintomas", ""))

        while True:
            dibujar(fila, seleccion, pos + 1, len(cola), aviso)
            aviso = ""

            try:
                entrada = input("Opcion> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo sin guardar el post actual.")
                return 0

            minuscula = entrada.lower()

            if minuscula == "q":
                print("\nProgreso guardado. Saliendo.")
                return 0

            if entrada == "" or minuscula == "a":
                antes = snapshot(fila)
                aplicar_guardado(fila, seleccion)
                despues = snapshot(fila)

                try:
                    guardar_csv(campos, filas)
                    registrar("guardar", fila, antes, despues)
                except Exception as exc:
                    restaurar(fila, antes)
                    aviso = "ERROR al guardar: {}".format(exc)
                    continue

                historial.append(
                    {"fila": fila, "pos": pos, "antes": antes}
                )
                pos += 1
                aviso = "Guardado; avanzando al siguiente post."
                break

            if entrada == "9":
                antes = snapshot(fila)
                aplicar_dudoso(fila)
                despues = snapshot(fila)

                try:
                    guardar_csv(campos, filas)
                    registrar("dudoso", fila, antes, despues)
                except Exception as exc:
                    restaurar(fila, antes)
                    aviso = "ERROR al marcar dudoso: {}".format(exc)
                    continue

                historial.append(
                    {"fila": fila, "pos": pos, "antes": antes}
                )
                pos += 1
                aviso = "Marcado como dudoso; avanzando."
                break

            if minuscula == "s":
                pos += 1
                aviso = "Post saltado."
                break

            if minuscula == "u":
                if not historial:
                    aviso = "No hay acciones para deshacer."
                    continue

                accion = historial.pop()
                fila_objetivo = accion["fila"]
                antes_deshacer = snapshot(fila_objetivo)
                restaurar(fila_objetivo, accion["antes"])
                despues_deshacer = snapshot(fila_objetivo)

                try:
                    guardar_csv(campos, filas)
                    registrar(
                        "deshacer",
                        fila_objetivo,
                        antes_deshacer,
                        despues_deshacer,
                    )
                except Exception as exc:
                    restaurar(fila_objetivo, antes_deshacer)
                    historial.append(accion)
                    aviso = "ERROR al deshacer: {}".format(exc)
                    continue

                pos = accion["pos"]
                aviso = "Deshecho; snapshot previo restaurado."
                break

            if entrada == "0":
                seleccion.clear()
                aviso = "Seleccion limpiada."
                continue

            if entrada.startswith("+") or entrada.startswith("-"):
                operador = entrada[0]
                numeros = parsear_numeros(entrada[1:])

                if numeros is None:
                    aviso = "Entrada no valida. Ejemplos: +14, -7, 1 7 14."
                    continue

                if operador == "+":
                    seleccion.update(numeros)
                    aviso = "Agregado(s): {}.".format(
                        formatear_gold(numeros)
                    )
                else:
                    seleccion.difference_update(numeros)
                    aviso = "Quitado(s): {}.".format(
                        formatear_gold(numeros)
                    )

                continue

            numeros = parsear_numeros(entrada)

            if numeros is None:
                aviso = (
                    "Entrada no valida. Use numeros 1-21, "
                    "0, 9, a, s, u o q."
                )
                continue

            for numero in numeros:
                if numero in seleccion:
                    seleccion.remove(numero)
                else:
                    seleccion.add(numero)

            aviso = "Sintomas alternados: {}.".format(
                formatear_gold(seleccion) if seleccion else "(ninguno)"
            )

    limpiar_pantalla()
    print("Revision terminada. No quedan posts pendientes en esta sesion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())