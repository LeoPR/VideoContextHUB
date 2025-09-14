#!/usr/bin/env python3
# coding: utf-8
"""
extract_diarization.py

Diarização com Resemblyzer + KMeans/Agglomerative usando transcript com palavras (word timestamps).

Agora centraliza leitura de áudio em utils.audio_io.AudioReader / load_wav_as_numpy.
Suporta extração de embeddings em modo streaming por janelas (seek+read) e opções de
diagnóstico/clustering:
- --show-scores: imprime silhouette_scores e inertias (quando aplicável)
- --cluster-method: escolha entre "kmeans" (padrão) e "agglo" (AgglomerativeClustering com distância cosine)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from utils.audio_io import AudioReader, load_wav_as_numpy

# Defaults (serão sobrescritos pelo config.json se presente)
DEFAULTS = {
    "diarization": {
        "window_size": 1.0,
        "hop": 0.5,
        "min_window_dur": 0.25,
        "merge_gap": 0.5,
        "num_speakers": None,     # None => estimar automaticamente
        "max_k": 10,
        "out_dir": ".cache/diarization",
        "hash_length": 16,
    },
    "output": {
        "base_dir": ".cache",
        "audio_subdir": "audio",
        "transcripts_subdir": "transcripts",
        "hash_length": 16,
        "json_indent": 2,
    },
}


def read_config(path: Optional[Path]) -> Dict:
    cfg_path = path or Path("config.json")
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
            return cfg
    except Exception:
        return {}


def merge_params(config: Dict, cli_args: argparse.Namespace) -> Dict:
    params = {
        "diarization": DEFAULTS["diarization"].copy(),
        "output": DEFAULTS["output"].copy(),
    }

    cfg_diar = (config.get("diarization") or {})
    cfg_out = (config.get("output") or config.get("io") or {})

    params["diarization"].update(cfg_diar)
    params["output"].update(cfg_out)

    # CLI overrides
    if getattr(cli_args, "window_size", None) is not None:
        params["diarization"]["window_size"] = float(cli_args.window_size)
    if getattr(cli_args, "hop", None) is not None:
        params["diarization"]["hop"] = float(cli_args.hop)
    if getattr(cli_args, "min_window_dur", None) is not None:
        params["diarization"]["min_window_dur"] = float(cli_args.min_window_dur)
    if getattr(cli_args, "merge_gap", None) is not None:
        params["diarization"]["merge_gap"] = float(cli_args.merge_gap)
    if getattr(cli_args, "num_speakers", None) is not None:
        params["diarization"]["num_speakers"] = int(cli_args.num_speakers)
    if getattr(cli_args, "max_k", None) is not None:
        params["diarization"]["max_k"] = int(cli_args.max_k)
    if getattr(cli_args, "out_dir", None) is not None:
        params["diarization"]["out_dir"] = str(cli_args.out_dir)
    if getattr(cli_args, "hash_length", None) is not None:
        params["diarization"]["hash_length"] = int(cli_args.hash_length)

    params["diarization"]["window_size"] = float(params["diarization"]["window_size"])
    params["diarization"]["hop"] = float(params["diarization"]["hop"])
    params["diarization"]["min_window_dur"] = float(params["diarization"]["min_window_dur"])
    params["diarization"]["merge_gap"] = float(params["diarization"]["merge_gap"])
    params["diarization"]["max_k"] = int(params["diarization"].get("max_k", 10))
    return params


def sha256_of_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_transcript_words(transcript_path: Path) -> List[Dict]:
    with transcript_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    words: List[Dict] = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({"start": float(w["start"]), "end": float(w["end"]), "word": (w.get("word") or "").strip()})
    if not words:
        raise ValueError("Transcript não contém palavras (words) com timestamps. Habilite word_timestamps no transcritor.")
    return words


def make_windows(duration: float, window_size: float, hop: float) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    starts = np.arange(0.0, max(0.0, duration - 1e-8), hop)
    windows: List[Tuple[float, float]] = []
    for s in starts:
        s = float(s)
        e = s + window_size
        if s < duration:
            windows.append((s, min(e, duration)))
    if not windows:
        windows = [(0.0, min(window_size, duration))]
    return windows


def embed_windows(
    wav: np.ndarray,
    sr: int,
    windows: List[Tuple[float, float]],
    min_window_dur: float,
    encoder: Optional[VoiceEncoder] = None,
    show_progress: bool = True,
) -> Tuple[List[Optional[np.ndarray]], List[Tuple[float, float]]]:
    """
    Compatibilidade: extrai embeddings a partir de um array numpy já carregado (mantido para compatibilidade).
    """
    if encoder is None:
        encoder = VoiceEncoder()

    embs: List[Optional[np.ndarray]] = []
    aligned_windows: List[Tuple[float, float]] = []
    it = tqdm(windows, desc="Extraindo embeddings") if show_progress else windows
    for (ws, we) in it:
        si = int(max(0, math.floor(ws * sr)))
        ei = int(min(len(wav), math.ceil(we * sr)))
        segment = wav[si:ei]
        if len(segment) < (min_window_dur * sr):
            embs.append(None)
            aligned_windows.append((ws, we))
            continue
        try:
            emb = encoder.embed_utterance(segment.astype(np.float32, copy=False))
            embs.append(emb)
        except Exception:
            embs.append(None)
        aligned_windows.append((ws, we))
    return embs, aligned_windows


def embed_windows_streaming(
    reader: AudioReader,
    sr: int,
    windows: List[Tuple[float, float]],
    min_window_dur: float,
    encoder: Optional[VoiceEncoder] = None,
    show_progress: bool = True,
) -> Tuple[List[Optional[np.ndarray]], List[Tuple[float, float]]]:
    """
    Extrai embeddings diretamente do arquivo usando AudioReader (seek+read por janela).
    - reader: AudioReader já aberto (context manager ativo).
    - sr: samplerate (inteiro) — também disponível em reader.info(), mas passado por conveniência.
    - windows: lista de (start_s, end_s) a serem extraídas.
    Retorna lista de embeddings (None quando janela curta/erro) e aligned windows.
    """
    if encoder is None:
        encoder = VoiceEncoder()

    embs: List[Optional[np.ndarray]] = []
    aligned_windows: List[Tuple[float, float]] = []
    it = tqdm(windows, desc="Extraindo embeddings") if show_progress else windows
    total_frames = int(getattr(reader, "frames", 0) or 0)
    for (ws, we) in it:
        si = int(max(0, math.floor(ws * sr)))
        ei = int(min(total_frames, math.ceil(we * sr))) if total_frames > 0 else int(math.ceil(we * sr))
        num_frames = max(0, ei - si)
        segment = reader.read_frames(si, num_frames)
        if len(segment) < (min_window_dur * sr):
            embs.append(None)
            aligned_windows.append((ws, we))
            continue
        try:
            emb = encoder.embed_utterance(segment.astype(np.float32, copy=False))
            embs.append(emb)
        except Exception:
            embs.append(None)
        aligned_windows.append((ws, we))
    return embs, aligned_windows


def cluster_embeddings(
    embs: List[Optional[np.ndarray]],
    k: int,
    method: str = "kmeans",
    random_state: int = 0,
):
    """
    Agrupa embeddings (lista que pode conter None) usando o método solicitado.
    Retorna (labels, model). labels tem mesmo tamanho de embs, com -1 para entradas None.
    method: "kmeans" ou "agglo"
    """
    valid = [e for e in embs if e is not None]
    if len(valid) == 0:
        raise RuntimeError("Nenhum embedding válido extraído.")
    X = np.stack(valid, axis=0)
    labels_out: List[int] = []
    if method == "kmeans":
        km = KMeans(n_clusters=max(1, min(k, len(valid))), random_state=random_state)
        km.fit(X)
        it = iter(km.labels_)
        for e in embs:
            if e is None:
                labels_out.append(-1)
            else:
                labels_out.append(int(next(it)))
        return labels_out, km
    elif method == "agglo":
        # AgglomerativeClustering com distância cosine:
        # calcula matriz de distâncias full e usa affinity='precomputed' com linkage='average'
        n = X.shape[0]
        # pairwise_distances retorna valores no intervalo [0, 2] para cosine; treat as distance
        D = pairwise_distances(X, metric="cosine")
        # Se k > n, ajustar
        k_eff = max(1, min(k, n))
        ac = AgglomerativeClustering(n_clusters=k_eff, affinity="precomputed", linkage="average")
        labels_dense = ac.fit_predict(D)
        it = iter(labels_dense)
        for e in embs:
            if e is None:
                labels_out.append(-1)
            else:
                labels_out.append(int(next(it)))
        return labels_out, ac
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def cluster_embeddings_kmeans(embs: List[Optional[np.ndarray]], k: int, random_state: int = 0):
    # Compat wrapper (mantido para compatibilidade interna se necessário)
    return cluster_embeddings(embs, k, method="kmeans", random_state=random_state)


def assign_speaker_to_words(words: List[Dict], windows: List[Tuple[float, float]], labels: List[int]) -> List[Dict]:
    centers = np.array([(ws + we) / 2.0 for (ws, we) in windows], dtype=np.float32) if windows else np.array([])
    words_out: List[Dict] = []
    for w in words:
        ws = float(w["start"])
        we = float(w["end"])
        overlapping: List[int] = []
        for i, (jws, jwe) in enumerate(windows):
            lab = labels[i]
            if lab == -1:
                continue
            if (jws < we) and (jwe > ws):
                overlapping.append(lab)
        if overlapping:
            chosen = int(Counter(overlapping).most_common(1)[0][0])
        else:
            if len(centers) == 0:
                chosen = 0
            else:
                mid = (ws + we) / 2.0
                idx = int(np.argmin(np.abs(centers - mid)))
                lab = labels[idx]
                if lab == -1:
                    valids = [L for L in labels if L != -1]
                    chosen = int(valids[0]) if valids else 0
                else:
                    chosen = int(lab)
        words_out.append({"start": ws, "end": we, "word": w.get("word", ""), "speaker": f"SPEAKER_{chosen + 1}"})
    return words_out


def merge_words_into_segments(words_with_spk: List[Dict], max_gap: float = 0.5) -> List[Dict]:
    if not words_with_spk:
        return []
    diarized_segments: List[Dict] = []
    cur = {
        "start": words_with_spk[0]["start"],
        "end": words_with_spk[0]["end"],
        "speaker": words_with_spk[0]["speaker"],
        "words": [words_with_spk[0]["word"]],
    }
    for w in words_with_spk[1:]:
        same = (w["speaker"] == cur["speaker"])
        gap = w["start"] - cur["end"]
        if same and gap <= max_gap:
            cur["end"] = w["end"]
            cur["words"].append(w["word"])
        else:
            cur["text"] = " ".join(cur["words"]).strip()
            diarized_segments.append(cur)
            cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "words": [w["word"]]}
    cur["text"] = " ".join(cur["words"]).strip()
    diarized_segments.append(cur)
    return diarized_segments


def estimate_num_speakers_auto(
    embs: List[Optional[np.ndarray]],
    max_k: int = 10,
    random_state: int = 0,
    cluster_method: str = "kmeans",
) -> Dict:
    """
    Estima k automaticamente testando valores de 2..max_k (ou 1..max_k) usando silhouette (preferencial)
    e calcula inertias quando aplicável (kmeans). Retorna dicionário com 'k', 'method', 'silhouette_scores', 'inertias'.
    Esta versão aceita cluster_method para usar o mesmo algoritmo na estimação.
    """
    valid = [e for e in embs if e is not None]
    n = len(valid)
    result = {"k": 1, "method": "insufficient_embeddings", "silhouette_scores": None, "inertias": None}
    if n == 0:
        return result
    max_k_eff = min(max_k, n)
    if n == 1 or max_k_eff <= 1:
        result["k"] = 1
        result["method"] = "single_embedding"
        return result

    X = np.stack(valid, axis=0)
    silhouette_scores = {}
    inertias = {}

    # Testa k de 2..max_k_eff
    for k in range(2, max_k_eff + 1):
        try:
            if cluster_method == "kmeans":
                km = KMeans(n_clusters=k, random_state=random_state)
                labels = km.fit_predict(X)
                # inertia disponível
                inertias[k] = float(km.inertia_)
                if len(set(labels)) > 1 and len(set(labels)) < len(X):
                    sc = float(silhouette_score(X, labels))
                    silhouette_scores[k] = sc
            elif cluster_method == "agglo":
                # Agglo com cosine: cluster sobre matriz de distâncias precomputada
                D = pairwise_distances(X, metric="cosine")
                ac = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
                labels = ac.fit_predict(D)
                if len(set(labels)) > 1 and len(set(labels)) < len(X):
                    sc = float(silhouette_score(X, labels, metric="cosine"))
                    silhouette_scores[k] = sc
                # inertia não aplicável para agglo
            else:
                # fallback para kmeans se método desconhecido
                km = KMeans(n_clusters=k, random_state=random_state)
                labels = km.fit_predict(X)
                inertias[k] = float(km.inertia_)
                if len(set(labels)) > 1 and len(set(labels)) < len(X):
                    sc = float(silhouette_score(X, labels))
                    silhouette_scores[k] = sc
        except Exception:
            continue

    if silhouette_scores:
        best_k = max(silhouette_scores.keys(), key=lambda kk: silhouette_scores[kk])
        result["k"] = int(best_k)
        result["method"] = "auto-silhouette"
        result["silhouette_scores"] = silhouette_scores
        result["inertias"] = inertias if inertias else None
        return result

    # Se silhouette não deu, tenta inércia/elbow (apenas para kmeans; para agglo retornamos inertias possivelmente vazio)
    if cluster_method == "kmeans":
        inertias_all = {}
        for k in range(1, max_k_eff + 1):
            try:
                km = KMeans(n_clusters=k, random_state=random_state)
                km.fit(X)
                inertias_all[k] = float(km.inertia_)
            except Exception:
                continue
        result["inertias"] = inertias_all
        if not inertias_all:
            result["k"] = 1
            result["method"] = "fallback-single"
            return result

        ks = sorted(inertias_all.keys())
        vals = [inertias_all[k] for k in ks]
        diffs = [vals[i - 1] - vals[i] for i in range(1, len(vals))]
        if len(diffs) < 2:
            chosen = min(2, max_k_eff)
            result["k"] = int(chosen)
            result["method"] = "inertia-fallback"
            return result
        sec = [diffs[i - 1] - diffs[i] for i in range(1, len(diffs))]
        idx = int(np.argmax(sec))
        chosen_k = ks[idx + 1] if (idx + 1) < len(ks) else ks[-1]
        result["k"] = int(max(1, chosen_k))
        result["method"] = "auto-inertia-elbow"
        return result
    else:
        # Para agglo sem silhouette, fallback simples
        result["k"] = 1
        result["method"] = "fallback-single-agglo"
        result["silhouette_scores"] = silhouette_scores if silhouette_scores else None
        result["inertias"] = None
        return result


def human_readable_size(num: int, suffix: str = "B") -> str:
    n = float(num)
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}{suffix}"
        n /= 1024.0
    return f"{n:.1f}P{suffix}"


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Diarização com Resemblyzer + KMeans/Agglo (estimativa automática de speakers)")
    p.add_argument("--config", default=None, help="caminho para config.json (opcional). Se ausente, tenta ./config.json")
    p.add_argument("--audio-wav", required=True, help="Caminho para WAV (normalizado pelo pipeline)")
    p.add_argument("--transcript", required=True, help="Caminho para transcript JSON com words")
    p.add_argument("--num-speakers", type=int, default=None, help="Override manual do número de speakers (se omitido, estima automaticamente)")
    p.add_argument("--max-k", type=int, default=None, help="Máximo K a testar na estimação automática (override do config)")
    p.add_argument("--window-size", type=float, default=None)
    p.add_argument("--hop", type=float, default=None)
    p.add_argument("--min-window-dur", type=float, default=None)
    p.add_argument("--merge-gap", type=float, default=None)
    p.add_argument("--out-dir", default=None, help="Diretório de saída (override do config)")
    p.add_argument("--hash-length", type=int, default=None)
    # Novas flags
    p.add_argument("--show-scores", action="store_true", help="Imprime silhouette_scores e inertias para cada k testado (diagnóstico)")
    p.add_argument("--cluster-method", choices=["kmeans", "agglo"], default="kmeans", help="Método de clustering a usar: kmeans (padrão) ou agglo (Agglomerative com distância cosine)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    cfg_path = Path(args.config) if args.config else None
    config = read_config(cfg_path)
    params = merge_params(config, args)
    diar_cfg = params["diarization"]
    out_dir = Path(diar_cfg["out_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio_wav)
    transcript_path = Path(args.transcript)

    if not audio_path.exists():
        raise FileNotFoundError(f"Áudio WAV não encontrado: {audio_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript JSON não encontrado: {transcript_path}")

    # Abre AudioReader e usa leitura por janelas (streaming) para extrair embeddings
    with AudioReader(audio_path) as reader:
        info = reader.info()
        sr = int(info.get("samplerate", 0))
        duration = float(info.get("duration", 0.0))
        print(f"[diar] WAV: {audio_path}  sr={sr}  duração={duration:.2f}s  tamanho={human_readable_size(audio_path.stat().st_size)}")

        words = load_transcript_words(transcript_path)
        print(f"[diar] Transcript: {len(words)} palavras com timestamps.")

        window_size = float(diar_cfg["window_size"])
        hop = float(diar_cfg["hop"])
        min_window_dur = float(diar_cfg["min_window_dur"])
        merge_gap = float(diar_cfg["merge_gap"])
        max_k = int(diar_cfg.get("max_k", 10))

        # CLI overrides
        if getattr(args, "window_size", None) is not None:
            window_size = float(args.window_size)
        if getattr(args, "hop", None) is not None:
            hop = float(args.hop)
        if getattr(args, "min_window_dur", None) is not None:
            min_window_dur = float(args.min_window_dur)
        if getattr(args, "merge_gap", None) is not None:
            merge_gap = float(args.merge_gap)
        if getattr(args, "max_k", None) is not None:
            max_k = int(args.max_k)

        windows = make_windows(duration, window_size=window_size, hop=hop)
        print(f"[diar] Gerando {len(windows)} janelas (window_size={window_size}, hop={hop})")
        encoder = VoiceEncoder()

        # Usa leitura por janelas para não carregar todo o WAV na memória
        embs, aligned_windows = embed_windows_streaming(reader, sr, windows, min_window_dur=min_window_dur, encoder=encoder, show_progress=True)

    # Estimação / clustering
    method_info = {"method": None, "k": None, "details": None}
    cluster_method = getattr(args, "cluster_method", "kmeans")
    manual_k = params["diarization"].get("num_speakers", None)
    if getattr(args, "num_speakers", None) is not None:
        manual_k = int(args.num_speakers)

    if manual_k is not None:
        k = int(manual_k)
        method_info["method"] = "manual"
        method_info["k"] = k
        method_info["details"] = {"note": "valor especificado via config/CLI"}
        print(f"[diar] Usando número de speakers manual: k={k}")
        labels, _km = cluster_embeddings(embs, k, method=cluster_method)
    else:
        est = estimate_num_speakers_auto(embs, max_k=max_k, random_state=0, cluster_method=cluster_method)
        k = int(est.get("k", 1))
        method_info["method"] = est.get("method", "auto")
        method_info["k"] = k
        method_info["details"] = {
            "silhouette_scores": est.get("silhouette_scores"),
            "inertias": est.get("inertias"),
        }
        print(f"[diar] Estimativa automática: método={method_info['method']}  k={k}")

        # Se --show-scores foi pedido, imprime as métricas testadas
        if getattr(args, "show_scores", False):
            print("[diar] Scores testados (k -> silhouette, inertia):")
            ss = est.get("silhouette_scores") or {}
            inert = est.get("inertias") or {}
            ks_sorted = sorted(set(list(ss.keys()) + list(inert.keys())))
            for kk in ks_sorted:
                sc = ss.get(kk, None)
                iv = inert.get(kk, None)
                sc_str = f"{sc:.4f}" if sc is not None else "n/a"
                iv_str = f"{iv:.4f}" if iv is not None else ("n/a" if cluster_method == "agglo" else "n/a")
                print(f"  k={kk}: silhouette={sc_str}  inertia={iv_str}")

        labels, _km = cluster_embeddings(embs, k, method=cluster_method)

    valid_count = sum(1 for e in embs if e is not None)
    print(f"[diar] Embeddings válidos: {valid_count}/{len(embs)}  -> clustering com k={k}")

    words_with_spk = assign_speaker_to_words(words, aligned_windows, labels)
    counts = Counter([w["speaker"] for w in words_with_spk])
    print("[diar] Contagem por speaker (palavras):")
    for s, c in counts.items():
        print(f"  {s}: {c}")

    diarized_segments = merge_words_into_segments(words_with_spk, max_gap=merge_gap)
    print(f"[diar] Segmentos diarizados: {len(diarized_segments)}")

    audio_sha = sha256_of_file(audio_path)
    short = audio_sha[:max(1, int(diar_cfg.get("hash_length", 16)))]
    out_json = out_dir / f"{short}-diarization.json"

    model_meta = {
        "name": "resemblyzer+clustering",
        "embedding_model": "Resemblyzer VoiceEncoder",
        "params": {
            "num_speakers": k,
            "window_size": window_size,
            "hop": hop,
            "min_window_dur": min_window_dur,
            "merge_gap": merge_gap,
            "max_k_tested": max_k,
            "cluster_method": cluster_method,
        },
    }

    payload = {
        "audio": str(audio_path),
        "transcript": str(transcript_path),
        "duration": duration,
        "model": model_meta,
        "estimation": {
            "method": method_info["method"],
            "k": method_info["k"],
            "details": method_info["details"],
        },
        "words": words_with_spk,
        "segments": diarized_segments,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=int(DEFAULTS["output"].get("json_indent", 2)))

    print(f"[diar] JSON salvo em: {out_json.resolve()}")


if __name__ == "__main__":
    main()