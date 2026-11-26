#!/usr/bin/env python3
r"""Construye un léxico de jerga trans desde el corpus, para el guided masking.

Ranking por KEYNESS contra una referencia de inglés general (Norvig count_1w):
log2(frec relativa del corpus / frec relativa de la referencia) como tamaño de
efecto y G2 de Dunning como significancia. Las palabras comunes del inglés
(start, months, years...) quedan fuera por construcción; lo que sube es lo que
este corpus usa muchísimo más que el inglés general (jerga).

Subcomandos
-----------
candidatos  Dos pasadas por el corpus (frecuencias + subreddits/contextos) y
            escribe candidatos_lexico_trans.csv para REVISIÓN MANUAL (columna
            "incluir"). Requiere --referencia.

aplicar     Lee candidatos_lexico_trans.csv, toma las filas con incluir en
            {1, x, X, si, sí, yes} y escribe lexico_trans.txt (solo las
            marcadas; el notebook ya trae las semillas base). Si ya existe,
            guarda .bak y fusiona.

auto        Se NEGARÁ a correr sin la referencia (lista sin revisar).

Ejemplos (desde mlm/ en WSL)
----------------------------
    python3 construir_lexico_trans.py candidatos --max-docs 100000
    python3 construir_lexico_trans.py candidatos
    python3 construir_lexico_trans.py aplicar
"""
import argparse
import collections
import html
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODELO = "citiusLTL/DisorBERT"

# semillas ORIGINALES (todas): solo para EXCLUIR de los candidatos
SEMILLAS_ORIGINALES = (
    "hrt t4t deadname deadnamed deadnaming euphoria binder binding packing packer "
    "tucking stealth clocked clock chaser boymode girlmode transmasc transfem "
    "nonbinary enby afab amab cis cisgender ftm mtf testosterone estrogen "
    "estradiol spironolactone spiro blockers transition transitioning detransition "
    "misgender pronouns passing egg ffs gatekeeping"
).split()

# semillas ESTRICTAS: solo para el grupo S de la columna informativa z_semilla
SEMILLAS_ESTRICTAS = (
    "hrt t4t deadname deadnamed deadnaming estradiol estrogen testosterone "
    "spironolactone spiro mtf ftm afab amab enby transmasc transfem boymode "
    "girlmode packer binder tucking"
).split()

# lista base de depresión del notebook: se EXCLUYE de los candidatos
DEPRESION = (
    "abandon abandoned abused afraid agony alone angry anguish anxiety anxious "
    "apathy ashamed awful bad betrayed bitter blame blamed blue broken burden cry "
    "crying dead death defeated depressed depressing depression despair desperate "
    "destroyed die died disappointment disgusted dislike dread dysphoria dysphoric "
    "dysmorphia empty enraged exhausted failure failed failing fake fear fearful "
    "flashback friendless frustrated furious grief guilty hate hated helpless "
    "hopeless hopelessness hurt ignored inferior insomnia invisible irritable "
    "jealous joyless kill lonely lose loser lost low meaningless misgendered "
    "misgendering miserable mood numb obsessed overwhelmed pain painful panic "
    "paranoid pathetic pointless punished regret rejected restless ruined sad "
    "sadness scared selfharm shame shamed sick silence sleepless stressed stupid "
    "suffer suffering suicide suicidal tired trapped trash transphobia transphobic "
    "ugly uncomfortable unhappy unwanted useless victim weak weep weird worthless "
    "worthlessness wrong"
).split()

STOPWORDS = set(
    "a about above after again all also am an and any are as at be because been "
    "before being below between both but by can could did do does doing down "
    "during each even few for from further had has have having he her hers here "
    "him his how i if in into is it its just like made make many me more most "
    "much my no nor not now of off on once only or other our out over own same "
    "she should so some such than that the their them then there these they this "
    "those through to too under until up very was we were what when where which "
    "while who why will with would you your yours".split()
)
CONTRACCIONES = {"didn", "isn", "don", "ve", "ll", "re", "doesn", "wasn",
                 "couldn", "wouldn", "shouldn", "aren", "hasn", "haven"}

# vocabulario de plataforma y registro informal de Reddit: se EXCLUYE fijo
PLATAFORMA = set(
    "subreddit subreddits reddit redditors redditor asktransgender asktg tldr "
    "tumblr youtube facebook okcupid nsfw throwaway lurker lurking rant ranting "
    "venting vent nbsp idk tbh irl tmi wtf ymmv ugh meh haha yay hey okay texted "
    "kinda sorta gonna dunno whatnot legit ish anyways cannot hadn weren wasn ive "
    "shes isnt didnt wont cant dont wasnt soooo sooo caitlyn jenner suporn "
    "x200b instagram tiktok twitter snapchat grindr gofundme pinterest whatsapp "
    "spotify minecraft chatgpt onlyfans covid groupon faceapp emoji youtuber "
    "youtubers downvote downvoted upvotes reddits goodrx incel rlly yknow "
    "sooooo soooooo idek idfk idrk tysm halp uwu bruh yeet lowkey obvs smth hiii "
    "heyo yalls aint bestie rambly uncomfy thicc tiddies".split())

# limpieza EXACTA del notebook
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_SUB = re.compile(r"\br/\w+|\bu/\w+", re.I)
RE_MD = re.compile(r"[*`~]{2,}|_{2,}")
RE_CTRL = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
RE_ESP = re.compile(r"\s+")
RE_PALABRA = re.compile(r"[a-z0-9]+")

LARGO_LISTA = 3000     # shortlist que pasa a la segunda pasada


def limpiar(s):
    s = unicodedata.normalize("NFKC", s); s = html.unescape(s)
    s = RE_URL.sub(" ", s); s = RE_SUB.sub(" ", s)
    s = s.replace("[deleted]", " ").replace("[removed]", " ")
    s = RE_MD.sub(" ", s); s = RE_CTRL.sub(" ", s)
    return RE_ESP.sub(" ", s).strip()


def localizar_referencia(nombre):
    """Busca la referencia en el cwd y en vocab/; None si no está."""
    for ruta in (Path(nombre), Path("vocab") / Path(nombre).name):
        if ruta.exists():
            return ruta
    return None


def cargar_referencia(ruta):
    """Lee count_1w (palabra TAB conteo). Ignora líneas mal formadas."""
    conteos, malformadas = {}, 0
    with open(ruta, encoding="utf-8", errors="replace") as f:
        for linea in f:
            partes = linea.strip().split()          # TAB o espacios
            if len(partes) != 2:
                if partes:
                    malformadas += 1
                continue
            try:
                n = int(partes[1])
            except ValueError:
                malformadas += 1
                continue
            if n > 0:
                conteos[partes[0].lower()] = conteos.get(partes[0].lower(), 0) + n
    return conteos, malformadas


def leer_lotes(ruta, max_docs):
    """Generador por lotes de (subreddit, texto_limpio). Nunca carga todo."""
    pf = pq.ParquetFile(ruta)
    restantes = max_docs
    for lote in pf.iter_batches(batch_size=50_000, columns=["title", "text", "subreddit"]):
        tits = lote.column("title").to_pylist()
        corps = lote.column("text").to_pylist()
        subs = lote.column("subreddit").to_pylist()
        docs = []
        for s, t, c in zip(subs, tits, corps):
            if restantes is not None and restantes <= 0:
                break
            docs.append(((s or "").lower(), limpiar((t or "") + " " + (c or ""))))
            if restantes is not None:
                restantes -= 1
        if docs:
            yield docs
        if restantes is not None and restantes <= 0:
            return


def _xlog(x, den):
    """x * log(x/den) con guarda: 0 donde no esté definido."""
    x = np.asarray(x, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = (x > 0) & (den > 0)
    salida = np.zeros_like(x)
    salida[ok] = x[ok] * np.log(x[ok] / den[ok])
    return salida


def calcular_candidatos(args):
    t0 = time.time()
    ruta = Path(args.corpus)
    assert ruta.exists(), f"no encuentro {ruta}"

    ruta_ref = localizar_referencia(args.referencia)
    if ruta_ref is None:
        raise SystemExit(
            f"no encuentro la referencia de inglés '{args.referencia}' "
            f"(ni en el cwd ni en vocab/).\nConsíguela con:\n"
            f"  curl -o {args.referencia} https://norvig.com/ngrams/count_1w.txt\n"
            f"o baja count_1w.txt y renómbralo {args.referencia}.")
    print(f"referencia: {ruta_ref}")

    ya = set()
    if Path("lexico_trans.txt").exists():
        ya = {p.strip().lower()
              for p in Path("lexico_trans.txt").read_text(encoding="utf-8").split()
              if p.strip()}
        print(f"excluyendo {len(ya)} palabras ya aceptadas de lexico_trans.txt")

    excluidas = set(SEMILLAS_ORIGINALES) | set(DEPRESION) | ya | CONTRACCIONES | PLATAFORMA
    estrictas = set(SEMILLAS_ESTRICTAS)

    # ── pasada 1: frecuencias + grupos de semillas estrictas ──────────────
    frec_total = collections.Counter()
    docfreq_s = collections.Counter()
    docfreq_n = collections.Counter()
    docs_s = docs_n = procesados = 0
    for docs in leer_lotes(ruta, args.max_docs):
        for _, texto in docs:
            palabras = RE_PALABRA.findall(texto.lower())
            frec_total.update(palabras)
            unicas = set(palabras)
            if unicas & estrictas:
                docs_s += 1
                docfreq_s.update(unicas)
            else:
                docs_n += 1
                docfreq_n.update(unicas)
        procesados += len(docs)
        if procesados % 200_000 == 0:
            print(f"  pasada 1 · {procesados:,} posts ({time.time() - t0:.0f} s)",
                  flush=True)
    print(f"pasada 1 lista: {procesados:,} posts · S {docs_s:,} · N {docs_n:,} · "
          f"vocabulario del corpus {len(frec_total):,} "
          f"({time.time() - t0:.0f} s)")

    # ── referencia de inglés general ─────────────────────────────────────
    conteos, malformadas = cargar_referencia(ruta_ref)
    n_ref = float(sum(conteos.values()))
    min_conteo = min(conteos.values())
    print(f"referencia: {len(conteos):,} palabras · total {n_ref:,.0f} conteos · "
          f"conteo mínimo {min_conteo:,} · líneas mal formadas {malformadas:,}")

    faltantes = sum(1 for w in frec_total if w not in conteos)
    print(f"palabras del vocabulario del corpus SIN entrada en la referencia: "
          f"{faltantes:,} de {len(frec_total):,}")

    # ── filtros de forma + keyness (tamaño del efecto) ────────────────────
    base = [w for w in frec_total
            if frec_total[w] >= args.min_freq
            and w not in excluidas and w not in STOPWORDS
            and len(w) >= 3 and not w[0].isdigit()]
    print(f"con frec >= {args.min_freq:,} y filtros de forma: {len(base):,} palabras")

    f = np.array([frec_total[w] for w in base], dtype=float)
    n_corpus = float(sum(frec_total.values()))
    c = np.array([conteos.get(w, 0) for w in base], dtype=float)
    en_ref = c > 0
    c_eff = np.maximum(c, min_conteo)              # pseudoconteo si no está
    log2_razon = np.log2((f / n_corpus) / (c_eff / n_ref))

    # G2 de Dunning (log-likelihood): 2 * sum(observado * ln(observado/esperado))
    e1 = n_corpus * (f + c_eff) / (n_corpus + n_ref)
    e2 = n_ref * (f + c_eff) / (n_corpus + n_ref)
    g2 = 2 * (_xlog(f, e1)
              + _xlog(n_corpus - f, n_corpus - e1)
              + _xlog(c_eff, e2)
              + _xlog(n_ref - c_eff, n_ref - e2))

    # candidato: efecto grande; NUNCA se ordena por z
    pasa = log2_razon >= args.min_log2
    idx = np.nonzero(pasa)[0]
    print(f"con log2_razon >= {args.min_log2} (el umbral filtra el registro "
          f"informal): {len(idx):,} palabras")

    # orden: log2_razon desc, desempate por frecuencia
    orden = idx[np.lexsort((f[idx], log2_razon[idx]))[::-1]]
    corto = orden[:LARGO_LISTA]
    shortlist = [base[i] for i in corto]
    log2_sl = {base[i]: float(log2_razon[i]) for i in corto}
    g2_sl = {base[i]: float(g2[i]) for i in corto}
    ref_sl = {base[i]: bool(en_ref[i]) for i in corto}
    print(f"shortlist para la segunda pasada: {len(shortlist):,}")

    # z de las semillas estrictas: solo columna informativa
    a = np.array([float(docfreq_s.get(w, 0)) for w in shortlist])
    b = np.array([float(docfreq_n.get(w, 0)) for w in shortlist])
    alpha0 = 1000.0
    a0 = float(sum(docfreq_s.values()))
    b0 = float(sum(docfreq_n.values()))
    p_pool = (a + b) / (a0 + b0)
    al = alpha0 * p_pool
    var = (1 / np.maximum(a + al, 1e-12)
           + 1 / np.maximum(a0 - a + alpha0 - al, 1e-12)
           + 1 / np.maximum(b + al, 1e-12)
           + 1 / np.maximum(b0 - b + alpha0 - al, 1e-12))
    delta = (np.log((a + al) / np.maximum(a0 - a + alpha0 - al, 1e-12))
             - np.log((b + al) / np.maximum(b0 - b + alpha0 - al, 1e-12)))
    z_sl = {w: float(d / math.sqrt(v)) for w, d, v in zip(shortlist, delta, var)}

    # ── segunda pasada: subreddits + un contexto por palabra ─────────────
    t1 = time.time()
    sub_por = {w: set() for w in shortlist}
    corto_set = set(shortlist)
    contextos, faltan = {}, set(shortlist)
    procesados = 0
    for docs in leer_lotes(ruta, args.max_docs):
        for sub, texto in docs:
            tocadas = corto_set.intersection(RE_PALABRA.findall(texto.lower()))
            if not tocadas:
                continue
            for w in tocadas:
                sub_por[w].add(sub)
            if faltan:
                for w in (tocadas & faltan):
                    m = re.search(r"\b" + re.escape(w) + r"\b", texto.lower())
                    if not m:
                        continue
                    i = m.start()
                    trozo = re.sub(r"\s+", " ",
                                   texto[max(0, i - 40):i + len(w) + 40]).strip()
                    contextos[w] = (("… " if i > 40 else "") + trozo
                                    + (" …" if i + len(w) + 40 < len(texto) else ""))
                    faltan.discard(w)
        procesados += len(docs)
        if procesados % 200_000 == 0:
            print(f"  pasada 2 · {procesados:,} posts · contextos {len(contextos):,} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(f"pasada 2 lista: contextos {len(contextos):,} de {len(shortlist):,} "
          f"({time.time() - t1:.0f} s)")

    filas = []
    for w in shortlist:
        if len(sub_por[w]) < args.min_subreddits:
            continue
        filas.append(dict(
            palabra=w,
            log2_razon=round(log2_sl[w], 2),
            g2=round(g2_sl[w], 1),
            frec_total=int(frec_total[w]),
            frec_rel_corpus=round(frec_total[w] / n_corpus, 8),
            en_referencia=ref_sl[w],
            z_semilla=round(z_sl[w], 2),
            docfreq_semilla=int(docfreq_s.get(w, 0)),
            docfreq_resto=int(docfreq_n.get(w, 0)),
            n_subreddits=len(sub_por[w])))
    filas.sort(key=lambda r: (-r["log2_razon"], -r["frec_total"]))
    filas = filas[:args.top]
    print(f"candidatos finales: {len(filas):,} "
          f"({sum(1 for r in filas if not r['en_referencia']):,} fuera de la referencia)")

    # piezas según el tokenizador de DisorBERT (red, luego caché, luego nada)
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(MODELO)
        except Exception:
            tok = AutoTokenizer.from_pretrained(MODELO, local_files_only=True)
            print("(tokenizador cargado desde la caché local: sin conexión)")
        for r in filas:
            piezas = tok.tokenize(r["palabra"])
            r["n_piezas"] = len(piezas)
            r["piezas"] = " ".join(piezas)
    except Exception:
        print("(!) sin tokenizador disponible (ni red ni caché): "
              "columnas de piezas vacías")
        for r in filas:
            r["n_piezas"] = ""
            r["piezas"] = ""
    for r in filas:
        r["contexto"] = contextos.get(r["palabra"], "")
        r["incluir"] = ""

    return pd.DataFrame(filas, columns=[
        "palabra", "log2_razon", "g2", "frec_total", "frec_rel_corpus",
        "en_referencia", "z_semilla", "docfreq_semilla", "docfreq_resto",
        "n_subreddits", "n_piezas", "piezas", "contexto", "incluir"])


def cmd_candidatos(args):
    t0 = time.time()
    df = calcular_candidatos(args)
    ruta = Path(args.salida)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\nescrito {ruta} · {len(df)} filas · {(time.time() - t0) / 60:.1f} min")
    try:
        import resource
        print(f"memoria máxima aprox: "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MiB")
    except Exception:
        pass
    if len(df):
        vista = df.head(40).copy()
        vista["contexto"] = vista["contexto"].astype(str).str.slice(0, 60)
        print("\n--- primeros 40 candidatos (orden: log2_razon desc) ---")
        print(vista.to_string(index=False))


def _marcado(v):
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return s in {"1", "x", "si", "sí", "yes"}


def cmd_aplicar(args):
    ruta_c = Path(args.candidatos)
    assert ruta_c.exists(), f"no encuentro {ruta_c}"
    df = pd.read_csv(ruta_c, encoding="utf-8-sig", dtype={"incluir": str})
    sel = sorted({str(p).strip().lower()
                  for p, m in zip(df["palabra"], df["incluir"]) if _marcado(m)})
    if not sel:
        print("no hay palabras marcadas en 'incluir': no escribo nada")
        return
    ruta = Path(args.salida)
    habia = []
    if ruta.exists():
        contenido = ruta.read_text(encoding="utf-8")
        if contenido.strip():
            bak = ruta.with_suffix(ruta.suffix + ".bak")
            bak.write_text(contenido, encoding="utf-8")
            print(f"respaldo: {bak}")
            habia = [p.strip().lower() for p in contenido.splitlines() if p.strip()]
    nuevas = [p for p in sel if p not in set(habia)]
    total = sorted(set(habia) | set(sel))
    ruta.write_text("\n".join(total) + "\n", encoding="utf-8")
    print(f"había {len(habia)} · marcadas ahora {len(sel)} · se añaden "
          f"{len(nuevas)} · total {len(total)}")
    print(f"escrito: {ruta}")


def cmd_auto(args):
    if localizar_referencia(args.referencia) is None:
        raise SystemExit(
            f"auto se niega a correr sin la referencia '{args.referencia}'.\n"
            f"Consíguela con:\n"
            f"  curl -o {args.referencia} https://norvig.com/ngrams/count_1w.txt")
    print("(!) auto marca SIN revisión manual: la lista no ha sido validada "
          "por nadie. Úsalo solo para pruebas o como punto de partida.\n")
    cmd_candidatos(args)
    ruta_c = Path(args.salida)
    df = pd.read_csv(ruta_c, encoding="utf-8-sig", dtype={"incluir": str})
    df["incluir"] = ""
    validos = df["n_subreddits"] >= 5
    a_marcar = df[validos].head(args.n).index
    df.loc[a_marcar, "incluir"] = "1"
    df.to_csv(ruta_c, index=False, encoding="utf-8-sig")
    print(f"marcadas automáticamente (n_subreddits >= 5): {len(a_marcar)}")
    ap = argparse.Namespace(**vars(args))
    ap.candidatos = str(ruta_c)
    ap.salida = "lexico_trans.txt"
    cmd_aplicar(ap)


def main():
    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument("--corpus", default="mlm_corpus.parquet")
    comunes.add_argument("--max-docs", type=int, default=None,
                         help="límite de posts para pruebas: toma las PRIMERAS "
                              "N filas del parquet (NO es una muestra aleatoria; "
                              "una submuestra puede no cubrir los 7 subreddits)")
    comunes.add_argument("--referencia", default="referencia_en.txt",
                         help="count_1w.txt de Norvig (palabra<TAB>conteo)")
    comunes.add_argument("--min-freq", type=int, default=200,
                         help="frecuencia mínima en el corpus")
    comunes.add_argument("--min-log2", type=float, default=4.0,
                         help="log2-ratio mínimo contra la referencia "
                              "(debajo de ~4 domina el registro informal)")
    comunes.add_argument("--min-subreddits", type=int, default=3,
                         help="mínimo de subreddits (de 7) donde aparece")
    comunes.add_argument("--top", type=int, default=600)

    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="comando", required=True)

    pc = sub.add_parser("candidatos", parents=[comunes],
                        help="genera candidatos_lexico_trans.csv")
    pc.add_argument("--salida", default="candidatos_lexico_trans.csv")
    pc.set_defaults(accion=cmd_candidatos)

    pa = sub.add_parser("aplicar", help="escribe lexico_trans.txt con lo marcado")
    pa.add_argument("--candidatos", default="candidatos_lexico_trans.csv")
    pa.add_argument("--salida", default="lexico_trans.txt")
    pa.set_defaults(accion=cmd_aplicar)

    pu = sub.add_parser("auto", parents=[comunes],
                        help="candidatos + marcar N automáticamente + aplicar "
                             "(requiere la referencia)")
    pu.add_argument("--n", type=int, default=150)
    pu.add_argument("--salida", default="candidatos_lexico_trans.csv")
    pu.set_defaults(accion=cmd_auto)

    args = p.parse_args()
    args.accion(args)


if __name__ == "__main__":
    main()
