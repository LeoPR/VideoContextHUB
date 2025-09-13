#!/usr/bin/env python3
"""
main.py

Lê config.json, prepara um WAV compatível (PCM16) para Whisper/diarizador:
- Se a entrada já for WAV compatível, retorna o próprio caminho.
- Caso contrário, gera/usa um WAV em cache com nome determinístico (SHA-256) dentro de <output.dir>/cache/audio.

Sem ffmpeg. Usa AudioExtractor (torchaudio + PyAV) do extract_audio.py.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import torchaudio  # usado para validar WAV compatível
from extract_audio import AudioExtractor, AudioExtractionError


def load_config(config_path: str | Path = "config.json") -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"config.json não encontrado em: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_wav_compatible(path: Path, target_sr: int, target_channels: int) -> bool:
    """
    Verifica se o arquivo WAV está exatamente no formato esperado:
    - PCM16 (quando detectável)
    - sample_rate == target_sr
    - num_channels == target_channels
    """
    try:
        info = torchaudio.info(str(path))
        sr = int(info.sample_rate) if info.sample_rate is not None else None
        ch = int(getattr(info, "num_channels", None)) if getattr(info, "num_channels", None) is not None else None
        bps = int(getattr(info, "bits_per_sample", -1)) if getattr(info, "bits_per_sample", None) is not None else None
        enc = getattr(info, "encoding", None) or getattr(info, "format", None) or ""
        enc = enc.upper()

        if sr != target_sr or ch != target_channels:
            return False

        # Preferir PCM16. Quando bits_per_sample não estiver disponível, aceitar se encoding indica PCM/PCM_S.
        is_pcm16 = (bps == 16) or ("PCM_S" in enc or "PCM" in enc)
        return bool(is_pcm16)
    except Exception:
        return False


def cache_path_for(media_input: Path, base_output_dir: Path, target_sr: int, target_channels: int) -> Path:
    """
    Gera um caminho de cache determinístico:
    <base_output_dir>/cache/audio/<sha256_16>-<sr>hz-<ch>ch.wav

    O hash é baseado em:
    - caminho absoluto
    - timestamp de modificação
    - tamanho do arquivo
    - parâmetros de destino (sr, ch)
    Isso ajuda a invalidar o cache quando o arquivo muda.
    """
    cache_dir = base_output_dir / "cache" / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)

    abs_str = str(media_input.resolve())
    try:
        st = media_input.stat()
        fingerprint = f"{abs_str}|{st.st_mtime_ns}|{st.st_size}|{target_sr}|{target_channels}"
    except FileNotFoundError:
        fingerprint = f"{abs_str}|{target_sr}|{target_channels}"

    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    out_name = f"{digest}-{target_sr}hz-{target_channels}ch.wav"
    return cache_dir / out_name


def prepare_audio_from_config(cfg: dict) -> Path:
    """
    Lê as configurações necessárias e retorna o caminho do WAV pronto:
    - media.input_path (obrigatório)
    - output.dir (base para salvar o cache; padrão 'outputs' se ausente)
    - sample_rate/channels: utiliza os valores do bloco media.ffmpeg (apenas como fonte numérica), com defaults 16000/1.
    """
    media_input = Path(cfg["media"]["input_path"])
    base_output_dir = Path(cfg.get("output", {}).get("dir", "outputs"))

    ffmpeg_cfg = cfg.get("media", {}).get("ffmpeg", {}) or {}
    # Só usamos os números; não usamos ffmpeg.
    target_sr = int(ffmpeg_cfg.get("sample_rate", 16000))
    target_channels = int(ffmpeg_cfg.get("channels", 1))

    if not media_input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {media_input}")

    # Se já for WAV compatível, retorne o próprio arquivo
    if media_input.suffix.lower() == ".wav" and is_wav_compatible(media_input, target_sr, target_channels):
        print(f"[main] Entrada já é WAV compatível — usando o próprio arquivo: {media_input}")
        return media_input

    # Caso contrário, converta/normalize para o cache
    out_wav = cache_path_for(media_input, base_output_dir, target_sr, target_channels)
    extractor = AudioExtractor(sample_rate=target_sr, channels=target_channels)
    out_path = extractor.extract(str(media_input), str(out_wav), overwrite=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Lê config.json e gera/retorna WAV compatível (cache determinístico).")
    parser.add_argument("--config", default="config.json", help="caminho para o arquivo de configuração (padrão: config.json)")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        out_wav = prepare_audio_from_config(cfg)
        print(f"[main] Áudio pronto: {out_wav}")
    except (AudioExtractionError, FileNotFoundError, KeyError) as e:
        print(f"[main] Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()