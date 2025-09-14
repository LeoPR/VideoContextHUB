#!/usr/bin/env python3
# coding: utf-8
"""
extract_video_context.py (atualizado: requer dependências via requirements, imports diretos)

Uso exemplo:
  python extract_video_context.py --video "M:/.../example.mp4" --target-fps 0.5 --group phash --hash-threshold 8 --step 5 --rep-selection closest --save-frames --out-dir .cache/video_context --max-samples 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from utils.video_frames import VideoFrameDataset, FrameGrouper, FrameGroup, group_by_config
from utils.config import ConfigLoader
from PIL import Image  # requerido; falha se não instalado


def sha_short(s: str, length: int = 12) -> str:
    import hashlib

    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:max(4, length)]


def ensure_out_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_pil_image_from_numpy(arr, path: Path):
    im = Image.fromarray(arr.astype("uint8"), mode="RGB")
    im.save(str(path))


def build_summary_json(
    video_path: str,
    dataset_info: Dict,
    groups: List[FrameGroup],
    selected_frames_meta: List[Dict],
    out_path: Path,
):
    payload = {
        "video": str(video_path),
        "dataset_info": dataset_info,
        "num_groups": len(groups) if groups is not None else 0,
        "groups": [
            {
                "rep": g.rep,
                "time": g.time,
                "members": g.members,
                "rep_hash": int(g.rep_hash) if g.rep_hash is not None else None,
                "rep_score": g.rep_score,
                "method": g.method,
                "threshold_used": g.threshold_used,
            }
            for g in (groups or [])
        ],
        "selected_frames": selected_frames_meta,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Extract video context (dry-run) com FrameGrouper unificado e ConfigLoader")
    p.add_argument("--config", default="config.json", help="caminho para config.json (procura media.input_path)")
    p.add_argument("--video", default=None, help="caminho para vídeo (override do config.json)")
    p.add_argument("--target-fps", type=float, default=None, help="FPS alvo para amostragem (ex: 0.5 -> 1 frame a cada 2s)")
    p.add_argument("--group", choices=["none", "ahash", "phash", "hist"], default=None, help="método de agrupamento (none = sem agrupamento)")
    p.add_argument("--hash-threshold", type=float, default=None, help="limiar: para ahash/phash é Hamming bits (int), para hist é float 0..1 (L2 normed)")
    p.add_argument("--step", type=int, default=None, help="passo ao percorrer frames para agrupamento/amostragem (ex.: 5 pula amostras)")
    p.add_argument("--max-samples", type=int, default=None, help="limita número de amostras consideradas (útil para vídeos longos)")
    p.add_argument("--rep-selection", choices=["first", "median", "closest"], default=None, help="estratégia para selecionar representante do grupo")
    p.add_argument("--bins-per-channel", default=None, help="bins para hist (ex: '8,8,8')")
    p.add_argument("--save-frames", action="store_true", help="salva os frames representativos em out-dir")
    p.add_argument("--out-dir", default=None, help="diretório de saída para JSON e frames salvos")
    p.add_argument("--max-save", type=int, default=None, help="máximo de frames a salvar quando --save-frames (proteção)")
    # opções adicionais para integração futura com Ollama (mantidas para overrides)
    p.add_argument("--ollama-host", default=None, help="host do Ollama (override do config)")
    p.add_argument("--prefer-multi", type=bool, default=None, help="prefere enviar múltiplas imagens por requisição ao Ollama")
    p.add_argument("--pull-if-missing", type=bool, default=None, help="baixar modelo se faltar (override)")
    p.add_argument("--use-probe", type=bool, default=None, help="usar probe ativo para detectar capacidades do modelo")
    return p.parse_args()


def main():
    args = parse_args()

    # Carrega config via ConfigLoader e aplica overrides CLI
    loader = ConfigLoader.from_file(args.config)
    loader.apply_cli_overrides(args)
    ok, errs = loader.validate(raise_on_error=False)
    if not ok:
        print("[warn] validação do config retornou erros:")
        for e in errs:
            print("  -", e)
    cfg = loader.app_config

    # Determina caminho do vídeo
    if args.video:
        video_path = args.video
    else:
        video_path = cfg.media.input_path
    if not video_path:
        raise SystemExit("Caminho do vídeo não informado (use --video ou defina media.input_path em config.json)")

    video_p = Path(video_path)
    if not video_p.exists():
        raise SystemExit(f"Arquivo de vídeo não encontrado: {video_path}")

    # Out dir
    out_dir = Path(cfg.media.out_dir or args.out_dir or ".")
    ensure_out_dir(out_dir)

    # Apply CLI overrides for a few direct flags (if provided)
    if args.target_fps is not None:
        cfg.media.target_fps = args.target_fps
    if args.step is not None:
        cfg.media.step = args.step
    if args.max_samples is not None:
        cfg.media.max_samples = args.max_samples
    if args.group is not None:
        cfg.grouping.method = args.group
    if args.hash_threshold is not None:
        cfg.grouping.hash_threshold = args.hash_threshold
    if args.rep_selection is not None:
        cfg.grouping.rep_selection = args.rep_selection
    if args.bins_per_channel is not None:
        # normalize "8,8,8" -> (8,8,8)
        parts = [int(x.strip()) for x in str(args.bins_per_channel).split(",") if x.strip()]
        if len(parts) == 1:
            parts = parts * 3
        cfg.grouping.bins_per_channel = tuple(parts)

    print(f"[run] vídeo: {video_path}")
    print(f"[run] target_fps: {cfg.media.target_fps}  group: {cfg.grouping.method}  media.step: {cfg.media.step}  grouping.step: {cfg.grouping.step}  max_samples: {cfg.media.max_samples}")

    # Constrói dataset (return_pil=False para operar com numpy; facilita hashing)
    ds = VideoFrameDataset(video_p, target_fps=cfg.media.target_fps, return_pil=False)
    info = ds.info()
    print(f"[info] dataset: {info}")

    # Prepara índices a considerar (respeita media.step e max_samples)
    media_step = max(1, int(cfg.media.step))
    idxs = list(range(0, len(ds), media_step))
    if cfg.media.max_samples is not None:
        idxs = idxs[: max(1, int(cfg.media.max_samples))]
    if idxs:
        print(f"[info] amostras consideradas: {len(idxs)} (indices {idxs[0]}..{idxs[-1]})")
    else:
        print("[info] nenhum índice a considerar")

    groups: List[FrameGroup] = []
    selected_frames_meta: List[Dict] = []

    if cfg.grouping.method != "none":
        method = cfg.grouping.method
        # determina threshold default se não informado
        if cfg.grouping.hash_threshold is None:
            if method in ("ahash", "phash"):
                threshold = 8
            else:
                threshold = 0.12
        else:
            threshold = cfg.grouping.hash_threshold

        bins = tuple(cfg.grouping.bins_per_channel)

        print(f"[group] método={method} threshold={threshold} media.step={media_step} grouping.step={cfg.grouping.step} rep_selection={cfg.grouping.rep_selection} bins={bins if method=='hist' else 'N/A'}")
        # use group_by_config helper for consistency
        groups = group_by_config(
            ds,
            cfg.grouping,
            max_items=cfg.media.max_samples,
            progress=True,
        )
        print(f"[group] grupos formados: {len(groups)}")
        for g in groups:
            selected_frames_meta.append({"index": int(g.rep), "time": float(g.time), "members": g.members, "rep_score": g.rep_score, "method": g.method, "threshold_used": g.threshold_used})
    else:
        # sem agrupamento: seleciona diretamente os índices amostrados (idxs)
        print("[sample] sem agrupamento: usando amostragem direta")
        for i in idxs:
            item = ds[i]
            selected_frames_meta.append({"index": int(i), "time": float(item["time"]), "members": [int(i)], "method": None, "rep_score": None, "threshold_used": None})

    # Salva frames representativos se pedido (leva em conta cfg.media.save_frames)
    saved = []
    if cfg.media.save_frames:
        nsave = min(len(selected_frames_meta), int(cfg.media.max_save or args.max_save or len(selected_frames_meta)))
        print(f"[save] salvando até {nsave} frames representativos em {out_dir}")
        for k, meta in enumerate(selected_frames_meta[:nsave]):
            idx = int(meta["index"])
            try:
                item = ds[idx]  # retorno numpy
                frame_np = item["frame"]
                base = f"frame-{idx:06d}-t{meta['time']:.3f}"
                fname = f"{base}.jpg"
                fpath = out_dir / fname
                save_pil_image_from_numpy(frame_np, fpath)
                meta["saved_path"] = str(fpath)
                saved.append(str(fpath))
                if (k + 1) % 20 == 0:
                    print(f"[save] {k+1}/{nsave} salvo")
            except Exception as e:
                print(f"[erro] salvando frame idx={idx}: {e}")

    # Monta e salva JSON resumo
    ident = sha_short(str(video_p.resolve()))
    out_json = out_dir / f"{ident}-video-context-dryrun.json"
    build_summary_json(video_p, info, groups, selected_frames_meta, out_json)
    print(f"[out] JSON salvo em: {out_json.resolve()}")
    if saved:
        print(f"[out] frames salvos: {len(saved)} (ex: {saved[0]})")

    ds.close()
    print("[done] dry-run concluído.")


if __name__ == "__main__":
    main()