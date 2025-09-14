#!/usr/bin/env python3
# coding: utf-8
"""
cleanup.py

Limpeza simples dos temporários deste projeto.

O que ele considera como "temporários do projeto":
- No temp do sistema (tempfile.gettempdir()): arquivos "tmp-segs-*.ndjson"
- No diretório de transcripts (output.base_dir + output.transcripts_subdir ou .cache/transcripts): "tmp-final-*.json"

Dois comandos:
- check: mostra quantos arquivos e quanto espaço os temporários ocupam
- clean: remove esses arquivos temporários (apenas os do projeto)

Uso:
  python cleanup.py check --config config.json
  python cleanup.py clean --config config.json --yes

Observações:
- Não há API do Python para "pedir ao OS" que limpe a pasta temp. Portanto, este script apaga apenas os temporários que o nosso código criou (padrões acima).
- Mantido simples, sem opções adicionais.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Padrões de arquivos temporários usados pelo projeto
SYSTEM_TEMP_PATTERNS = ["tmp-segs-*.ndjson"]    # gerados por extract_v2t no temp do SO
OUT_DIR_PATTERNS    = ["tmp-final-*.json"]      # quase-finais no out_dir (transcripts)


@dataclass
class TempStats:
    count: int
    bytes: int
    files: List[Path]


class Cleaner:
    def __init__(self, config_path: Optional[str] = None):
        self.system_temp = Path(tempfile.gettempdir())
        self.out_dir = self._discover_out_dir(config_path)

    def _discover_out_dir(self, config_path: Optional[str]) -> Path:
        """
        Lê config.json (se existir) para descobrir output.base_dir + transcripts_subdir.
        Fallback: .cache/transcripts
        """
        default = Path(".cache") / "transcripts"
        if not config_path:
            return default

        cfg_path = Path(config_path)
        if not cfg_path.exists():
            return default

        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            out_cfg = cfg.get("output", {}) or {}
            base = Path(out_cfg.get("base_dir") or out_cfg.get("dir") or ".cache")
            trans = out_cfg.get("transcripts_subdir", "transcripts")
            return base / trans
        except Exception:
            return default

    def _collect(self, base: Path, patterns: List[str]) -> List[Path]:
        files: List[Path] = []
        for pat in patterns:
            for p in base.glob(pat):
                if p.is_file():
                    files.append(p)
        # remover duplicatas por caminho
        uniq = []
        seen = set()
        for p in files:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def _stats_for(self, files: List[Path]) -> TempStats:
        total = 0
        for p in files:
            try:
                total += p.stat().st_size
            except Exception:
                pass
        return TempStats(count=len(files), bytes=total, files=files)

    def scan(self) -> Dict[str, TempStats]:
        """
        Varre os locais de temporários do projeto e retorna estatísticas.
        """
        sys_files = self._collect(self.system_temp, SYSTEM_TEMP_PATTERNS)
        out_files = []
        if self.out_dir.exists():
            out_files = self._collect(self.out_dir, OUT_DIR_PATTERNS)

        return {
            "system_temp": self._stats_for(sys_files),
            "out_dir": self._stats_for(out_files),
        }

    def report(self) -> None:
        stats = self.scan()
        sys_s = stats["system_temp"]
        out_s = stats["out_dir"]
        total_bytes = sys_s.bytes + out_s.bytes
        total_count = sys_s.count + out_s.count

        print(f"Temporários do projeto")
        print(f"- Temp do sistema ({self.system_temp}): {sys_s.count} arquivos, {self._human(sys_s.bytes)}")
        print(f"- Transcripts tmp ({self.out_dir}): {out_s.count} arquivos, {self._human(out_s.bytes)}")
        print(f"TOTAL: {total_count} arquivos, {self._human(total_bytes)}")

    def clean(self) -> Dict[str, int]:
        """
        Remove arquivos temporários do projeto (apenas os padrões conhecidos).
        Retorna um dict com contadores removidos e bytes liberados.
        """
        stats = self.scan()
        removed_count = 0
        removed_bytes = 0

        # Remove do temp do sistema
        for p in stats["system_temp"].files:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            try:
                os.remove(p)
                removed_count += 1
                removed_bytes += size
                print(f"REMOVIDO: {p}")
            except Exception as e:
                print(f"ERRO removendo {p}: {e}")

        # Remove do out_dir (se existir)
        for p in stats["out_dir"].files:
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            try:
                os.remove(p)
                removed_count += 1
                removed_bytes += size
                print(f"REMOVIDO: {p}")
            except Exception as e:
                print(f"ERRO removendo {p}: {e}")

        print(f"Limpeza concluída: {removed_count} arquivos removidos, {self._human(removed_bytes)} liberados.")
        return {"removed_files": removed_count, "freed_bytes": removed_bytes}

    @staticmethod
    def _human(num: int, suffix: str = "B") -> str:
        n = float(num)
        for unit in ["", "K", "M", "G", "T", "P"]:
            if abs(n) < 1024.0:
                return f"{n:3.1f}{unit}{suffix}"
            n /= 1024.0
        return f"{n:.1f}P{suffix}"


def _parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Cleanup simples dos temporários do projeto")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="mostra quanto espaço os temporários ocupam")
    p_check.add_argument("--config", default=None, help="caminho para config.json (opcional)")

    p_clean = sub.add_parser("clean", help="remove os temporários do projeto")
    p_clean.add_argument("--config", default=None, help="caminho para config.json (opcional)")
    p_clean.add_argument("--yes", action="store_true", help="confirmar limpeza sem perguntar")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = _parse_args(argv)

    cleaner = Cleaner(config_path=args.config)

    if args.cmd == "check":
        cleaner.report()
        return

    if args.cmd == "clean":
        if not args.yes:
            print("Use --yes para confirmar a limpeza.")
            return
        cleaner.clean()
        return


if __name__ == "__main__":
    main()