#!/usr/bin/env python3
# coding: utf-8
"""
utils/video_frames.py

Implementações leves e coesas para:
- VideoFrameDataset: iterador/índice de frames amostrados de um vídeo.
- FrameGroup: estrutura para representar um grupo de frames e metadados.
- FrameGrouper.group_by_hash: agrupamento por ahash / phash / hist
- Integração direta com utils.config.GroupingConfig:
    - validate_grouping_config
    - group_by_config

Observações:
- Dependências (obrigatórias): opencv-python (cv2), numpy, pillow (PIL).
- Não há importações condicionais: se as libs não estiverem instaladas, ocorrerá ImportError.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2  # opencv-python (obrigatório)
import numpy as np  # numpy (obrigatório)
from PIL import Image  # pillow (obrigatório)

# Importamos GroupingConfig para integração final
from utils.config import GroupingConfig  # type: ignore


@dataclass
class FrameGroup:
    """
    Representa um grupo de frames.

    Campos:
      rep: índice representativo (int)
      time: tempo (segundos) do frame representativo
      members: lista de índices (ints) pertencentes ao grupo
      rep_hash: hash numérico/hex do representante (pode ser int ou str)
      rep_score: escore de representatividade (float) — menor é melhor quando distâncias
      method: método usado (ahash/phash/hist)
      threshold_used: limiar usado para formar o grupo
    """
    rep: int
    time: float
    members: List[int]
    rep_hash: Optional[Any] = None
    rep_score: Optional[float] = None
    method: Optional[str] = None
    threshold_used: Optional[float] = None


class VideoFrameDataset:
    """
    Dataset leve para acessar frames extraídos de um arquivo de vídeo.

    - target_fps: se None, usa todos os frames na taxa do vídeo.
    - return_pil: se True, retorna PIL.Image; se False, retorna numpy.ndarray (HWC, uint8).
    """

    def __init__(self, path: str, target_fps: Optional[float] = None, return_pil: bool = False):
        self.path = str(path)
        self.target_fps = target_fps
        self.return_pil = return_pil
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {self.path}")
        # metadata
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = (self.frame_count / self.fps) if (self.fps > 0 and self.frame_count > 0) else None
        # build sampled indices based on target_fps
        self._indices = self._build_indices()

    def _build_indices(self) -> List[int]:
        if self.frame_count <= 0:
            return []
        if not self.target_fps or self.target_fps <= 0 or not self.fps or self.fps <= 0:
            return list(range(self.frame_count))
        # sample every (fps/target_fps) frames
        step = max(1, int(round(self.fps / float(self.target_fps))))
        return list(range(0, self.frame_count, step))

    def info(self) -> Dict[str, Any]:
        return {"path": self.path, "fps": self.fps, "frame_count": self.frame_count, "duration": self.duration, "sampled": len(self._indices)}

    def __len__(self) -> int:
        return len(self._indices)

    def _read_frame_at(self, frame_idx: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Retorna (frame_numpy, time_seconds) para o índice absoluto frame_idx do vídeo.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, float(frame_idx) / max(1.0, self.fps)
        # cv2 retorna BGR; convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = float(frame_idx) / max(1.0, self.fps)
        return frame_rgb, t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        idx é índice relativo dentro dos índices amostrados (0..len(self)-1).
        Retorna dict: {"index": absolute_frame_index, "time": seconds, "frame": numpy_or_pil}
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        abs_idx = self._indices[idx]
        frame_np, t = self._read_frame_at(abs_idx)
        if frame_np is None:
            raise RuntimeError(f"Erro lendo frame {abs_idx}")
        if self.return_pil:
            pil = Image.fromarray(frame_np)
            return {"index": abs_idx, "time": t, "frame": pil}
        return {"index": abs_idx, "time": t, "frame": frame_np}

    def close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


class FrameGrouper:
    """
    Grupo e hashes básicos:
    - ahash (average hash)
    - phash (perceptual hash via DCT)
    - hist (RGB histogram L2 normalized)

    Algoritmo de agrupamento (greedy):
      para cada frame i:
        calcula hash/descritor
        tenta associar a um grupo existente (compara com rep_hash usando distância)
        se encontrado (dist <= threshold) insere; atualiza representante se necessário
        senão cria novo grupo com esse frame como rep

    Esse algoritmo é simples e determinístico; serve para prototipagem.
    """

    @staticmethod
    def _to_pil(img: np.ndarray) -> "Image.Image":
        return Image.fromarray(img)

    @staticmethod
    def _ahash(img: np.ndarray, hash_size: int = 8) -> int:
        """
        Average hash: redimensiona para hash_size x hash_size em grayscale, compara com média.
        Retorna int com bits do hash.
        """
        pil = FrameGrouper._to_pil(img).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        arr = np.array(pil, dtype=np.uint8)
        mean = arr.mean()
        bits = (arr > mean).reshape(-1)
        h = 0
        for b in bits:
            h = (h << 1) | int(bool(b))
        return h

    @staticmethod
    def _phash(img: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
        """
        Perceptual hash (pHash) implementation:
        - resize to (hash_size*hf, hash_size*hf)
        - convert to grayscale
        - compute 2D DCT, take top-left (hash_size x hash_size) low frequencies excluding DC
        - compare to median to build bits
        """
        pil = FrameGrouper._to_pil(img).convert("L")
        size = hash_size * highfreq_factor
        pil = pil.resize((size, size), Image.LANCZOS)
        arr = np.array(pil, dtype=np.float32)
        # DCT via cv2.dct when disponível
        try:
            dct = cv2.dct(arr)
        except Exception:
            dct = np.real(np.fft.fft2(arr))
        # take top-left block
        tl = dct[:hash_size, :hash_size]
        med = np.median(tl)
        bits = (tl > med).reshape(-1)
        h = 0
        for b in bits:
            h = (h << 1) | int(bool(b))
        return int(h)

    @staticmethod
    def _hist_descriptor(img: np.ndarray, bins_per_channel: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        """
        Calcula histograma concatenado e normalizado (L2).
        Retorna vetor 1D float.
        """
        arr = np.array(img, dtype=np.uint8)
        # arr shape H,W,3 (RGB)
        chans = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=bins_per_channel[ch], range=(0, 256), density=False)
            chans.append(h.astype(np.float32))
        vec = np.concatenate(chans, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        x = a ^ b
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")

    @staticmethod
    def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def group_by_hash(
        ds: VideoFrameDataset,
        method: str = "ahash",
        threshold: float = 8,
        step: int = 1,
        max_items: Optional[int] = None,
        rep_selection: str = "first",
        bins_per_channel: Tuple[int, int, int] = (8, 8, 8),
        progress: bool = False,
    ) -> List[FrameGroup]:
        """
        Implementação principal de agrupamento.

        Parâmetros:
          ds: VideoFrameDataset (amostras já definidas por target_fps)
          method: "ahash" | "phash" | "hist"
          threshold: int para ahash/phash (Hamming), float para hist (L2)
          step: percorre ds com step
          max_items: limitar número de itens considerados (após step)
          rep_selection: "first" | "median" | "closest"
          bins_per_channel: para hist
        """
        method = (method or "ahash").lower()
        if method not in ("ahash", "phash", "hist"):
            raise ValueError("method deve ser um de: ahash, phash, hist")

        # indices relativos em ds
        indices = list(range(0, len(ds), max(1, int(step))))
        if max_items is not None:
            indices = indices[: max(1, int(max_items))]
        groups: List[FrameGroup] = []
        reps_hashes: List[Any] = []
        reps_vectors: List[Any] = []  # store numeric descriptors for hist
        reps_idxs: List[int] = []

        total = len(indices)
        for pos, rel_idx in enumerate(indices):
            if progress and (pos % 50 == 0):
                print(f"[group] processando {pos}/{total}...")
            item = ds[rel_idx]
            frame = item["frame"]
            abs_idx = int(item["index"])
            t = float(item["time"])
            # compute descriptor/hash
            if method == "ahash":
                desc = FrameGrouper._ahash(frame)
            elif method == "phash":
                desc = FrameGrouper._phash(frame)
            else:  # hist
                desc = FrameGrouper._hist_descriptor(frame, bins_per_channel)

            matched = False
            # compare to existing reps
            for gi, rep_hash in enumerate(reps_hashes):
                if method in ("ahash", "phash"):
                    dist = FrameGrouper._hamming_distance(int(rep_hash), int(desc))
                    sim = dist
                    is_similar = dist <= int(threshold)
                else:
                    # hist: use L2 between descriptors
                    vec_rep = reps_vectors[gi]
                    dist = FrameGrouper._l2_distance(vec_rep, desc)
                    sim = dist
                    is_similar = dist <= float(threshold)
                if is_similar:
                    # add to group
                    g = groups[gi]
                    g.members.append(abs_idx)
                    # optionally update representative according to rep_selection
                    if rep_selection == "first":
                        pass  # keep existing rep
                    elif rep_selection == "median":
                        mid = len(g.members) // 2
                        g.rep = g.members[mid]
                        g.rep_hash = reps_hashes[gi]
                    elif rep_selection == "closest":
                        if method in ("ahash", "phash"):
                            cur_rep_hash = int(reps_hashes[gi])
                            if FrameGrouper._hamming_distance(cur_rep_hash, int(desc)) < FrameGrouper._hamming_distance(int(g.rep_hash or cur_rep_hash), int(desc)):
                                g.rep = abs_idx
                                g.rep_hash = desc
                        else:
                            cur_rep_vec = reps_vectors[gi]
                            if FrameGrouper._l2_distance(cur_rep_vec, desc) < (g.rep_score or 1e9):
                                g.rep = abs_idx
                                g.rep_hash = desc
                    matched = True
                    break

            if not matched:
                # create new group
                g = FrameGroup(rep=abs_idx, time=t, members=[abs_idx], rep_hash=desc, rep_score=0.0, method=method, threshold_used=threshold)
                groups.append(g)
                reps_hashes.append(desc if method in ("ahash", "phash") else None)
                if method == "hist":
                    reps_vectors.append(desc)
                else:
                    reps_vectors.append(None)
                reps_idxs.append(abs_idx)

        # compute simple rep_score for hist groups (avg dist to rep) to help downstream
        if method == "hist":
            for g_i, g in enumerate(groups):
                rep_vec = reps_vectors[g_i]
                scores = []
                for mem_abs in g.members:
                    try:
                        rel = ds._indices.index(mem_abs)
                        it = ds[rel]
                        vec = FrameGrouper._hist_descriptor(it["frame"], bins_per_channel)
                        scores.append(FrameGrouper._l2_distance(rep_vec, vec))
                    except Exception:
                        continue
                g.rep_score = float(sum(scores) / len(scores)) if scores else 0.0

        return groups


# -------------------------
# Integração com GroupingConfig
# -------------------------
from typing import List, Optional, Tuple  # noqa: E402


def validate_grouping_config(grouping: GroupingConfig) -> Tuple[bool, List[str]]:
    """
    Validação leve de consistência do GroupingConfig.
    Retorna (ok, errors).
    """
    errs: List[str] = []
    method = (grouping.method or "none").lower()
    if method not in ("none", "ahash", "phash", "hist"):
        errs.append(f"method inválido: {grouping.method}")
    if grouping.rep_selection not in ("first", "median", "closest"):
        errs.append(f"rep_selection inválido: {grouping.rep_selection}")
    b = grouping.bins_per_channel
    if not (isinstance(b, (tuple, list)) and len(b) == 3):
        errs.append(f"bins_per_channel inválido: {b}")
    else:
        try:
            for x in b:
                if int(x) <= 0:
                    raise ValueError()
        except Exception:
            errs.append(f"bins_per_channel precisa conter inteiros positivos: {b}")
    # threshold checks (loose)
    if grouping.hash_threshold is not None:
        if method in ("ahash", "phash"):
            try:
                _ = int(grouping.hash_threshold)
            except Exception:
                errs.append("hash_threshold deve ser inteiro para ahash/phash")
        elif method == "hist":
            try:
                v = float(grouping.hash_threshold)
                if not (0.0 <= v <= 1.0):
                    errs.append("hash_threshold para hist deve estar em 0.0..1.0 (L2 normalizado)")
            except Exception:
                errs.append("hash_threshold deve ser float para hist")
    ok = len(errs) == 0
    return ok, errs


def group_by_config(
    ds: VideoFrameDataset,
    grouping_config: GroupingConfig,
    max_items: Optional[int] = None,
    progress: bool = False,
) -> List[FrameGroup]:
    """
    Adaptador que interpreta GroupingConfig e chama FrameGrouper.group_by_hash.

    Regras:
    - Se grouping_config.method == "none" retorna lista vazia.
    - Define thresholds padrão se hash_threshold é None:
        - ahash/phash -> 8
        - hist -> 0.12
    - Converte types conforme método:
        - ahash/phash: threshold -> int
        - hist: threshold -> float (0..1)
    - Passa bins_per_channel diretamente para o agrupador (para hist).
    - Usa grouping_config.step como step (garante >=1).
    """
    if grouping_config is None:
        raise ValueError("grouping_config não pode ser None")

    method = (grouping_config.method or "none").lower()
    if method == "none":
        return []

    ok, errs = validate_grouping_config(grouping_config)
    if not ok:
        raise ValueError("GroupingConfig inválido: " + "; ".join(errs))

    # defaults
    if grouping_config.hash_threshold is None:
        threshold = 8 if method in ("ahash", "phash") else 0.12
    else:
        threshold = grouping_config.hash_threshold

    # coerce types
    if method in ("ahash", "phash"):
        try:
            threshold = int(threshold)
            if threshold < 0:
                raise ValueError("threshold negativo")
        except Exception as e:
            raise ValueError(f"Threshold inválido para {method}: {grouping_config.hash_threshold} ({e})")
    else:
        try:
            threshold = float(threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("threshold fora do intervalo 0..1")
        except Exception as e:
            raise ValueError(f"Threshold inválido para hist: {grouping_config.hash_threshold} ({e})")

    bins = tuple(grouping_config.bins_per_channel or (8, 8, 8))
    step = max(1, int(getattr(grouping_config, "step", 1)))
    rep_selection = grouping_config.rep_selection or "first"

    groups = FrameGrouper.group_by_hash(
        ds,
        method=method,
        threshold=threshold,
        step=step,
        max_items=max_items,
        rep_selection=rep_selection,
        bins_per_channel=bins,
        progress=progress,
    )
    return groups


__all__ = [
    "VideoFrameDataset",
    "FrameGrouper",
    "FrameGroup",
    "group_by_config",
    "validate_grouping_config",
]