#!/usr/bin/env python3
"""
main_v2.py

Fluxo integrado revisado para evitar duplicação de "cache" no caminho:
- Lê config.json
- Gera/obtém WAV compatível via AudioExtractor (extract_audio.py) e salva em <base_dir>/<audio_subdir>
- Transcreve em streaming via V2TExtractor (extract_v2t.py) e salva em <base_dir>/<transcripts_subdir>
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

from extract_audio import AudioExtractor, AudioExtractionError
from extract_v2t import V2TExtractor

import torchaudio


def load_config(config_path: str | Path = "config.json") -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"config.json não encontrado em: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_wav_compatible(path: Path, target_sr: int, target_channels: int) -> bool:
    try:
        info = torchaudio.info(str(path))
        sr = int(info.sample_rate) if info.sample_rate is not None else None
        ch = int(getattr(info, "num_channels", None)) if getattr(info, "num_channels", None) is not None else None
        bps = int(getattr(info, "bits_per_sample", -1)) if getattr(info, "bits_per_sample", None) is not None else None
        enc = getattr(info, "encoding", None) or getattr(info, "format", None) or ""
        enc = enc.upper()

        if sr != target_sr or ch != target_channels:
            return False

        is_pcm16 = (bps == 16) or ("PCM_S" in enc or "PCM" in enc)
        return bool(is_pcm16)
    except Exception:
        return False


def cache_path_for(media_input: Path, base_output_dir: Path, audio_subdir: str, target_sr: int, target_channels: int, hash_length: int = 16) -> Path:
    """
    Gera um caminho de cache determinístico para o WAV:
    <base_output_dir>/<audio_subdir>/<sha>-<sr>hz-<ch>ch.wav

    Usa hash baseado em caminho absoluto + tamanho + mtime para detectar mudanças.
    Não adiciona um "cache" extra — usa apenas o base_output_dir + audio_subdir.
    """
    cache_dir = base_output_dir / audio_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    abs_str = str(media_input.resolve())
    try:
        st = media_input.stat()
        fingerprint = f"{abs_str}|{st.st_mtime_ns}|{st.st_size}|{target_sr}|{target_channels}"
    except FileNotFoundError:
        fingerprint = f"{abs_str}|{target_sr}|{target_channels}"

    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:hash_length]
    out_name = f"{digest}-{target_sr}hz-{target_channels}ch.wav"
    return cache_dir / out_name


def prepare_audio_from_config(cfg: dict, overwrite: bool = True) -> Path:
    """
    Lê as configurações necessárias e retorna o caminho do WAV pronto:
    - media.input_path (obrigatório)
    - output.base_dir ou output.dir (compatibilidade) e output.audio_subdir
    - sample_rate/channels: utiliza os valores do bloco media.ffmpeg (apenas como valores numéricos)
    """
    media_input = Path(cfg["media"]["input_path"])

    # Compatibilidade: preferir output.base_dir, senão output.dir, senão ".cache"
    output_cfg = cfg.get("output", {}) or {}
    base_output_dir = Path(output_cfg.get("base_dir") or output_cfg.get("dir") or ".cache")
    audio_subdir = output_cfg.get("audio_subdir", "audio")

    ffmpeg_cfg = cfg.get("media", {}).get("ffmpeg", {}) or {}
    target_sr = int(ffmpeg_cfg.get("sample_rate", 16000))
    target_channels = int(ffmpeg_cfg.get("channels", 1))

    hash_length = int(output_cfg.get("hash_length", 16))

    if not media_input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {media_input}")

    # Se já for WAV compatível, retorne o próprio arquivo
    if media_input.suffix.lower() == ".wav" and is_wav_compatible(media_input, target_sr, target_channels):
        print(f"[main] Entrada já é WAV compatível — usando o próprio arquivo: {media_input}")
        return media_input

    # Caso contrário, converte/normalize para o cache (em base_output_dir / audio_subdir)
    out_wav = cache_path_for(media_input, base_output_dir, audio_subdir, target_sr, target_channels, hash_length=hash_length)
    extractor = AudioExtractor(sample_rate=target_sr, channels=target_channels)
    out_path = extractor.extract(str(media_input), str(out_wav), overwrite=overwrite)
    return out_path


def run_pipeline(config_path: str | Path = "config.json", overwrite_transcript: bool = False):
    cfg = load_config(config_path)

    # 1) Obter/gerar WAV compatível
    audio_wav = prepare_audio_from_config(cfg, overwrite=True)
    print(f"[main] WAV pronto: {audio_wav}")

    # 2) Configurações do Whisper a partir do config.json
    whisper_cfg = cfg.get("whisper", {}) or {}
    model_name = whisper_cfg.get("model", "small")
    device = whisper_cfg.get("device", None)  # faster_whisper pode autodetectar
    compute_type = whisper_cfg.get("compute_type", None)
    batch_size = int(whisper_cfg.get("batch_size", 8))
    beam_size = int(whisper_cfg.get("beam_size", 5))
    vad_filter = bool(whisper_cfg.get("vad_filter", True))
    language = whisper_cfg.get("language", None)
    condition_on_previous_text = bool(whisper_cfg.get("condition_on_previous_text", False))
    word_timestamps = bool(whisper_cfg.get("word_timestamps", True))

    # 3) Diretório base para salvar transcripts: use output.base_dir (ou output.dir) e subdir transcripts
    output_cfg = cfg.get("output", {}) or {}
    base_out = Path(output_cfg.get("base_dir") or output_cfg.get("dir") or ".cache")
    transcripts_dirname = output_cfg.get("transcripts_subdir", "transcripts")
    transcripts_dir = base_out / transcripts_dirname
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    hash_length = int(output_cfg.get("hash_length", 16))

    # 4) Inicializa V2TExtractor
    v2t = V2TExtractor(model_name=model_name, device=device, compute_type=compute_type, out_dir=transcripts_dir)

    # 5) Transcreve em streaming
    print("[main] Iniciando transcrição (streaming)...")
    try:
        gen = v2t.transcribe_file(
            audio_wav,
            batch_size=batch_size,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            overwrite=overwrite_transcript,
            on_segment=None,  # opcional: passar um callback para processar cada seg imediatamente
            hash_length=hash_length,
        )
        for seg in gen:
            # Aqui podemos encaminhar cada segmento à diarização; por enquanto apenas imprimimos
            print(json.dumps(seg, ensure_ascii=False))
        final_path = v2t.last_transcript_path
        print(f"[main] Transcrição salva em: {final_path}")
    except Exception as e:
        print(f"[main] Erro na transcrição: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description="Pipeline: extrair áudio (cache) e transcrever (streaming) com faster_whisper.")
    parser.add_argument("--config", default="config.json", help="caminho para config.json")
    parser.add_argument("--overwrite-transcript", action="store_true", help="sobrescrever transcript existente")
    args = parser.parse_args()

    try:
        run_pipeline(args.config, overwrite_transcript=args.overwrite_transcript)
    except (AudioExtractionError, FileNotFoundError, KeyError) as e:
        print(f"[main] Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()