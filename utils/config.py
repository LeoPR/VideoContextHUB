#!/usr/bin/env python3
# coding: utf-8
"""
utils/config.py

Carrega e valida config.json em um sistema de classes leve.
Objetivo: centralizar leitura/normalização/overrides de CLI e repassar configurações
tipadas para outras partes do sistema (VideoFrameDataset, FrameGrouper, OllamaClient).

Características:
- Dataclasses: MediaConfig, GroupingConfig, OllamaConfig, AppConfig
- ConfigLoader: métodos from_file, apply_cli_overrides, to_dict, save, validate
- Normaliza formatos comuns (bins_per_channel pode ser "8,8,8" ou [8,8,8])
- Sem dependências externas (usa stdlib)

Uso mínimo:
  from utils.config import ConfigLoader
  cfg = ConfigLoader.from_file("config.json")
  cfg.apply_cli_overrides(args)   # opcional, passa argparse.Namespace
  cfg.validate(raise_on_error=False)
  app_cfg = cfg.app_config        # AppConfig com campos tipados

Notas:
- Não altero o comportamento dos scripts existentes; ao integrar, troque leituras diretas de config.json
  por ConfigLoader.from_file(...).app_config.
- Mantenha mudanças pequenas e testáveis; se quiser que eu atualize extract_video_context.py para usar
  esse loader, aprove depois e eu gero apenas o script atualizado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _parse_bins_like(value: Union[str, List[int], Tuple[int, ...], None]) -> Tuple[int, int, int]:
    """
    Normaliza bins_per_channel que pode vir como:
      - "8,8,8"
      - "8"
      - [8,8,8]
      - 8
    Retorna tupla (r,g,b)
    """
    if value is None:
        return (8, 8, 8)
    if isinstance(value, (list, tuple)):
        nums = [int(x) for x in value]
        if len(nums) == 1:
            return (nums[0], nums[0], nums[0])
        if len(nums) >= 3:
            return (nums[0], nums[1], nums[2])
        # pad
        while len(nums) < 3:
            nums.append(nums[-1])
        return (nums[0], nums[1], nums[2])
    if isinstance(value, (int, float)):
        v = int(value)
        return (v, v, v)
    # assume string
    s = str(value).strip()
    if not s:
        return (8, 8, 8)
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        return (nums[0], nums[0], nums[0])
    if len(nums) >= 3:
        return (nums[0], nums[1], nums[2])
    while len(nums) < 3:
        nums.append(nums[-1])
    return (nums[0], nums[1], nums[2])


@dataclass
class MediaConfig:
    input_path: Optional[str] = None
    target_fps: Optional[float] = None
    step: int = 1
    max_samples: Optional[int] = None
    save_frames: bool = False
    out_dir: str = ".cache/video_context"
    max_save: int = 200

    def resolved_input(self) -> Optional[Path]:
        if self.input_path is None:
            return None
        return Path(self.input_path).expanduser().resolve()


@dataclass
class GroupingConfig:
    method: str = "none"  # none | ahash | phash | hist
    hash_threshold: Optional[Union[int, float]] = None
    rep_selection: str = "first"  # first | median | closest
    bins_per_channel: Tuple[int, int, int] = field(default_factory=lambda: (8, 8, 8))
    step: int = 1

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "GroupingConfig":
        if not d:
            return GroupingConfig()
        method = d.get("method", "none")
        ht = d.get("hash_threshold", None)
        rep = d.get("rep_selection", "first")
        bins_val = d.get("bins_per_channel", d.get("bins", (8, 8, 8)))
        bins = _parse_bins_like(bins_val)
        step = int(d.get("step", 1))
        # For hist threshold expect float, for ahash/phash expect int; leave as-is, higher-level code decides.
        return GroupingConfig(method=method, hash_threshold=ht, rep_selection=rep, bins_per_channel=bins, step=step)


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    prefer_multi: bool = True
    pull_if_missing: bool = True
    use_probe: bool = False
    # Optional lists of preferred models
    general_models: List[str] = field(default_factory=lambda: ["gemma3:12b", "qwen2-vl", "llama3.2-vision:11b"])
    ocr_models: List[str] = field(default_factory=lambda: ["minicpm-v", "qwen2-vl"])

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "OllamaConfig":
        if not d:
            return OllamaConfig()
        host = d.get("host", "http://localhost:11434")
        prefer_multi = bool(d.get("prefer_multi", True))
        pull_if_missing = bool(d.get("pull_if_missing", True))
        use_probe = bool(d.get("use_probe", False))
        general = d.get("general_models", d.get("models", [])) or []
        ocr = d.get("ocr_models", []) or []
        return OllamaConfig(
            host=host,
            prefer_multi=prefer_multi,
            pull_if_missing=pull_if_missing,
            use_probe=use_probe,
            general_models=list(general),
            ocr_models=list(ocr),
        )


@dataclass
class AppConfig:
    media: MediaConfig = field(default_factory=MediaConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


class ConfigLoader:
    """
    Loader/normalizer para config.json.

    Exemplos:
      loader = ConfigLoader.from_file("config.json")
      loader.apply_cli_overrides(args)   # args = argparse.Namespace
      loader.validate(raise_on_error=False)
      cfg: AppConfig = loader.app_config
    """

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        self.raw: Dict[str, Any] = raw or {}
        self.app_config: AppConfig = self._build_app_config(self.raw)

    @staticmethod
    def _build_app_config(raw: Dict[str, Any]) -> AppConfig:
        media_raw = raw.get("media", {}) or {}
        grouping_raw = raw.get("grouping", raw.get("group", {})) or {}
        ollama_raw = raw.get("ollama", {}) or {}

        media = MediaConfig(
            input_path=media_raw.get("input_path", None),
            target_fps=media_raw.get("target_fps", None),
            step=int(media_raw.get("step", 1)),
            max_samples=media_raw.get("max_samples", None),
            save_frames=bool(media_raw.get("save_frames", False)),
            out_dir=media_raw.get("out_dir", ".cache/video_context"),
            max_save=int(media_raw.get("max_save", 200)),
        )

        grouping = GroupingConfig.from_dict(grouping_raw)

        ollama = OllamaConfig.from_dict(ollama_raw)

        return AppConfig(media=media, grouping=grouping, ollama=ollama)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ConfigLoader":
        p = Path(path)
        if not p.exists():
            # retorna loader com defaults vazios (não falha aqui)
            return cls(raw={})
        try:
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}
        except Exception:
            raw = {}
        return cls(raw=raw)

    def apply_cli_overrides(self, args: Any) -> None:
        """
        Aplica overrides a partir de argparse.Namespace (ou objeto com atributos similares).
        Só sobrescreve campos presentes em args (não definidos como None).
        """
        # media overrides
        if hasattr(args, "video") and getattr(args, "video") is not None:
            self.app_config.media.input_path = str(getattr(args, "video"))
        if hasattr(args, "target_fps") and getattr(args, "target_fps") is not None:
            self.app_config.media.target_fps = float(getattr(args, "target_fps"))
        if hasattr(args, "step") and getattr(args, "step") is not None:
            self.app_config.media.step = int(getattr(args, "step"))
        if hasattr(args, "max_samples") and getattr(args, "max_samples") is not None:
            self.app_config.media.max_samples = int(getattr(args, "max_samples"))
        if hasattr(args, "save_frames") and getattr(args, "save_frames") is not None:
            self.app_config.media.save_frames = bool(getattr(args, "save_frames"))
        if hasattr(args, "out_dir") and getattr(args, "out_dir") is not None:
            self.app_config.media.out_dir = str(getattr(args, "out_dir"))
        if hasattr(args, "max_save") and getattr(args, "max_save") is not None:
            self.app_config.media.max_save = int(getattr(args, "max_save"))

        # grouping overrides
        if hasattr(args, "group") and getattr(args, "group") is not None:
            self.app_config.grouping.method = str(getattr(args, "group"))
        if hasattr(args, "hash_threshold") and getattr(args, "hash_threshold") is not None:
            # hash_threshold can be int or float depending on method; try to cast intelligently
            val = getattr(args, "hash_threshold")
            try:
                ival = int(val)
                self.app_config.grouping.hash_threshold = ival
            except Exception:
                try:
                    fval = float(val)
                    self.app_config.grouping.hash_threshold = fval
                except Exception:
                    self.app_config.grouping.hash_threshold = val  # keep raw
        if hasattr(args, "rep_selection") and getattr(args, "rep_selection") is not None:
            self.app_config.grouping.rep_selection = str(getattr(args, "rep_selection"))
        if hasattr(args, "bins_per_channel") and getattr(args, "bins_per_channel") is not None:
            bins_raw = getattr(args, "bins_per_channel")
            self.app_config.grouping.bins_per_channel = _parse_bins_like(bins_raw)
        if hasattr(args, "step") and getattr(args, "step") is not None:
            self.app_config.grouping.step = int(getattr(args, "step"))

        # ollama overrides
        if hasattr(args, "ollama_host") and getattr(args, "ollama_host") is not None:
            self.app_config.ollama.host = str(getattr(args, "ollama_host"))
        if hasattr(args, "prefer_multi") and getattr(args, "prefer_multi") is not None:
            self.app_config.ollama.prefer_multi = bool(getattr(args, "prefer_multi"))
        if hasattr(args, "pull_if_missing") and getattr(args, "pull_if_missing") is not None:
            self.app_config.ollama.pull_if_missing = bool(getattr(args, "pull_if_missing"))
        if hasattr(args, "use_probe") and getattr(args, "use_probe") is not None:
            self.app_config.ollama.use_probe = bool(getattr(args, "use_probe"))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa o AppConfig para dicionário pronto para JSON.
        """
        return {
            "media": asdict(self.app_config.media),
            "grouping": {
                **asdict(self.app_config.grouping),
                "bins_per_channel": list(self.app_config.grouping.bins_per_channel),
            },
            "ollama": asdict(self.app_config.ollama),
        }

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def validate(self, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
        """
        Valida heurística e retorna (ok, errors).
        Se raise_on_error=True, lança ValueError com as mensagens concatenadas em caso de erro.
        Validações feitas:
          - se media.input_path informado, checa existência do arquivo
          - grouping.method é um dos permitidos
          - grouping.rep_selection é válido
          - grouping.bins_per_channel é tupla de 3 inteiros positivos
          - ollama.host é uma string não vazia
        """
        errs: List[str] = []
        # media
        if self.app_config.media.input_path:
            p = Path(self.app_config.media.input_path).expanduser()
            if not p.exists():
                errs.append(f"media.input_path não existe: {self.app_config.media.input_path}")
        # grouping
        if self.app_config.grouping.method not in ("none", "ahash", "phash", "hist"):
            errs.append(f"grouping.method inválido: {self.app_config.grouping.method}")
        if self.app_config.grouping.rep_selection not in ("first", "median", "closest"):
            errs.append(f"grouping.rep_selection inválido: {self.app_config.grouping.rep_selection}")
        b = self.app_config.grouping.bins_per_channel
        if not (isinstance(b, tuple) or isinstance(b, list)) or len(b) != 3:
            errs.append(f"grouping.bins_per_channel inválido: {b}")
        else:
            try:
                for x in b:
                    if int(x) <= 0:
                        raise ValueError()
            except Exception:
                errs.append(f"grouping.bins_per_channel precisa conter inteiros positivos: {b}")
        # ollama
        if not self.app_config.ollama.host or not isinstance(self.app_config.ollama.host, str):
            errs.append("ollama.host inválido ou vazio")

        ok = len(errs) == 0
        if not ok and raise_on_error:
            raise ValueError("Config validation errors:\n" + "\n".join(errs))
        return ok, errs