#!/usr/bin/env python3
"""
extract_v2t.py

V2TExtractor — Transcrição streaming de arquivos de áudio para JSON usando faster_whisper,
salvando resultados em .cache/transcripts com nome determinístico baseado no SHA-256 do conteúdo.

Agora o JSON final inclui metadados do modelo usado:
  "model": { "name": ..., "compute_type": ..., "rank": ... }

Ao iniciar uma transcrição, se o arquivo já existir e overwrite=False, o extractor compara
o rank do modelo existente com o rank do modelo solicitado:
  - se existing_rank >= requested_rank -> reusa e reemite segmentos (nenhuma nova transcrição)
  - se existing_rank < requested_rank  -> re-transcreve e sobrescreve o arquivo
"""

from pathlib import Path
import hashlib
import json
import tempfile
import os
from typing import Optional, Callable, Generator, Dict, Any

from faster_whisper import WhisperModel, BatchedInferencePipeline


class V2TExtractor:
    def __init__(
        self,
        model_name: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        out_dir: Path | str = Path(".cache") / "transcripts",
    ):
        """
        Args:
            model_name: nome/caminho do modelo faster_whisper (ex: "small", "medium", "large-v3")
            device: "cuda" ou "cpu" ou None (faster_whisper detecta)
            compute_type: ex: "float16", "int8_float16", etc.
            out_dir: diretório base para salvar transcripts (default: .cache/transcripts)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Inicializa modelo e pipeline
        self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        self.pipeline = BatchedInferencePipeline(model=self.model)

        self.last_transcript_path: Optional[Path] = None

    # -------------------------
    # Utilitários
    # -------------------------
    @staticmethod
    def _sha256_file(path: Path, block_size: int = 65536) -> str:
        """Calcula SHA-256 do arquivo em streaming e retorna hex completo."""
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _model_rank(model_name: Optional[str]) -> int:
        """
        Atribui um rank numérico simples ao modelo com base no nome, usado para decidir
        se um novo modelo é 'melhor' que o anterior.
        Regras (padrão, pode ajustar conforme desejar):
          - tiny / tiny.* -> 0
          - small -> 1
          - medium -> 2
          - large -> 3
          - large-v2 / large-v3 -> 4
        Se model_name for None ou desconhecido, retorna -1.
        """
        if not model_name:
            return -1
        m = model_name.lower()
        if "large-v3" in m or "large_v3" in m or "largev3" in m:
            return 4
        if "large-v2" in m or "large_v2" in m or "largev2" in m:
            return 4
        if "large" in m:
            return 3
        if "medium" in m:
            return 2
        if "small" in m:
            return 1
        if "tiny" in m:
            return 0
        # fallback: se contém números maiores, dar rank maior
        if any(tok.isdigit() for tok in m.split("-") + m.split("_")):
            try:
                nums = [int(''.join(filter(str.isdigit, tok))) for tok in m.replace("_", "-").split("-") if any(ch.isdigit() for ch in tok)]
                if nums:
                    return min(4, max(0, max(nums)))
            except Exception:
                pass
        return -1

    def _transcript_path_for(self, audio_path: Path, sha_hex: str, hash_length: int = 16) -> Path:
        """Gera caminho final do transcript baseado no hash (usa primeiros hash_length hex)."""
        short = sha_hex[:hash_length]
        name = f"{short}-transcript.json"
        return self.out_dir / name

    # -------------------------
    # API principal
    # -------------------------
    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        batch_size: int = 8,
        beam_size: int = 5,
        vad_filter: bool = True,
        language: Optional[str] = "pt",
        condition_on_previous_text: bool = False,
        word_timestamps: bool = True,
        overwrite: bool = False,
        on_segment: Optional[Callable[[Dict[str, Any]], None]] = None,
        hash_length: int = 16,
    ) -> Generator[Dict[str, Any], None, Path]:
        """
        Stream de transcrição com lógica de reuso por modelo:
        - calcula hash do arquivo de áudio (conteúdo)
        - prepara arquivo de saída em out_dir/<shaN>-transcript.json
        - se já existir e overwrite=False, compara rank do modelo existente vs solicitado:
            - se existing_rank >= requested_rank: reemite segmentos do arquivo existente (sem transcrever)
            - se existing_rank < requested_rank: faz nova transcrição e sobrescreve
        - caso contrário, executa BatchedInferencePipeline.transcribe e:
            - escreve cada segmento como uma linha NDJSON num temp file
            - yield cada segmento (dict) para o chamador
        - Ao final, monta o JSON final lendo linhas NDJSON e escreve o arquivo final atômico,
          incluindo o bloco "model" com name/compute_type/rank.
        - Retorna (via StopIteration.value) o Path do arquivo salvo e também atualiza self.last_transcript_path
        """
        audio_p = Path(audio_path)
        if not audio_p.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_p}")

        # calcula hash do conteúdo (streaming)
        sha_full = self._sha256_file(audio_p)
        out_path = self._transcript_path_for(audio_p, sha_full, hash_length=hash_length)

        requested_rank = self._model_rank(self.model_name)

        # Se já existir e não sobrescrever, inspeciona o modelo salvo
        if out_path.exists() and not overwrite:
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                existing_model = data.get("model", {}) or {}
                existing_name = existing_model.get("name")
                existing_rank = int(existing_model.get("rank", -1))
            except Exception:
                existing_rank = -1
                existing_name = None

            # Se já foi gerado por modelo igual/maior, reemite e retorna
            if existing_rank >= requested_rank:
                # reemitir segmentos existentes
                with out_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    for seg in data.get("segments", []):
                        if on_segment:
                            try:
                                on_segment(seg)
                            except Exception:
                                pass
                        yield seg
                self.last_transcript_path = out_path
                return out_path
            else:
                # caso contrário, vai sobrescrever (re-transcrever)
                pass  # prossegue para a transcrição

        # prepara arquivos temporários no mesmo diretório de out_path
        tmp_ndjson_fd, tmp_ndjson_name = tempfile.mkstemp(prefix="tmp-segs-", suffix=".ndjson", dir=str(self.out_dir))
        os.close(tmp_ndjson_fd)
        tmp_ndjson_path = Path(tmp_ndjson_name)

        normalized_tmp_fd, normalized_tmp_name = tempfile.mkstemp(prefix="tmp-final-", suffix=".json", dir=str(self.out_dir))
        os.close(normalized_tmp_fd)
        normalized_tmp_path = Path(normalized_tmp_name)

        transcript_meta = {"language": None, "duration": None}

        try:
            # Abre NDJSON temp para escrita de segmentos
            with tmp_ndjson_path.open("w", encoding="utf-8") as seg_wf:
                # Executa a transcrição via pipeline
                segments_iter, info = self.pipeline.transcribe(
                    str(audio_p),
                    batch_size=batch_size,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    language=language,
                    condition_on_previous_text=condition_on_previous_text,
                    word_timestamps=word_timestamps,
                )

                # Extrai metadados do info
                try:
                    transcript_meta["language"] = getattr(info, "language", None) or None
                    transcript_meta["duration"] = float(getattr(info, "duration", 0.0) or 0.0)
                except Exception:
                    transcript_meta["language"] = None
                    transcript_meta["duration"] = None

                # Itera e escreve cada segmento como uma linha JSON (NDJSON), e yield
                for seg in segments_iter:
                    sdict = {
                        "start": float(getattr(seg, "start", seg.get("start") if isinstance(seg, dict) else 0.0)),
                        "end": float(getattr(seg, "end", seg.get("end") if isinstance(seg, dict) else 0.0)),
                        "text": (getattr(seg, "text", None) or seg.get("text") if isinstance(seg, dict) else "").strip(),
                        "words": [],
                    }

                    words = getattr(seg, "words", None) or (seg.get("words") if isinstance(seg, dict) else None) or []
                    for w in words:
                        if isinstance(w, dict):
                            wdict = {"start": float(w.get("start", 0.0)), "end": float(w.get("end", 0.0)), "word": w.get("word", "")}
                        else:
                            wdict = {"start": float(getattr(w, "start", 0.0)), "end": float(getattr(w, "end", 0.0)), "word": getattr(w, "word", "")}
                        sdict["words"].append(wdict)

                    # escreve NDJSON (uma linha por segmento)
                    seg_wf.write(json.dumps(sdict, ensure_ascii=False) + "\n")
                    seg_wf.flush()

                    # callback e yield
                    if on_segment:
                        try:
                            on_segment(sdict)
                        except Exception:
                            pass

                    yield sdict

            # Após terminar a transcrição, monta o JSON final lendo a NDJSON (streaming)
            model_meta = {
                "name": self.model_name,
                "compute_type": self.compute_type,
                "rank": self._model_rank(self.model_name),
            }

            with normalized_tmp_path.open("w", encoding="utf-8") as nf:
                # Escreve início do objeto com language/duration/model e abre array segments
                head = {
                    "language": transcript_meta.get("language"),
                    "duration": transcript_meta.get("duration"),
                    "model": model_meta,
                    "segments": []
                }
                # Escrevemos o cabeçalho sem ...segments para inserir itens manualmente, mantendo pretty format
                nf.write('{\n')
                nf.write(f'  "language": {json.dumps(head["language"], ensure_ascii=False)},\n')
                nf.write(f'  "duration": {json.dumps(head["duration"], ensure_ascii=False)},\n')
                nf.write(f'  "model": {json.dumps(head["model"], ensure_ascii=False)},\n')
                nf.write('  "segments": [\n')

                # reabrir e iterar NDJSON para escrever segmentos dentro do array
                first = True
                with tmp_ndjson_path.open("r", encoding="utf-8") as seg_rf:
                    for line in seg_rf:
                        line = line.strip()
                        if not line:
                            continue
                        if first:
                            nf.write("    " + line)
                            first = False
                        else:
                            nf.write(",\n    " + line)
                    nf.write("\n  ]\n}\n")  # fecha array e objeto

            # mover normalized_tmp para out_path de forma atômica
            if out_path.exists():
                out_path.unlink()
            os.replace(str(normalized_tmp_path), str(out_path))

            # limpar tmp ndjson
            try:
                if tmp_ndjson_path.exists():
                    tmp_ndjson_path.unlink()
            except Exception:
                pass

            self.last_transcript_path = out_path
            return out_path

        except Exception as e:
            # limpar arquivos temporários em caso de erro
            try:
                if tmp_ndjson_path.exists():
                    tmp_ndjson_path.unlink()
            except Exception:
                pass
            try:
                if normalized_tmp_path.exists():
                    normalized_tmp_path.unlink()
            except Exception:
                pass
            raise e

    # Conveniência: consome o generator e retorna o Path final
    def transcribe_and_wait(
        self,
        audio_path: str | Path,
        **kwargs,
    ) -> Path:
        """
        Conveniência: executa transcribe_file e consome todo o generator, retornando o Path final do transcript.
        """
        gen = self.transcribe_file(audio_path, **kwargs)
        for _ in gen:
            pass
        final_path = self.last_transcript_path
        if final_path is None:
            raise RuntimeError("Transcrição terminou sem produzir arquivo de saída.")
        return final_path