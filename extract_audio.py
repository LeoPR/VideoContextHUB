#!/usr/bin/env python3
"""
extract_audio.py

AudioExtractor — extrai e normaliza áudio preferindo torchaudio quando aplicável
e usando PyAV (av) para contêineres (mp4/mkv/mov/avi/webm/m4a), convertendo
apenas quando necessário para o formato ideal ao Whisper/diarizador:
- WAV PCM16
- Sample rate alvo (padrão 16000 Hz)
- Canais alvo (padrão 1 = mono)

Regras:
- Sem ffmpeg.
- Dependências obrigatórias: torch, torchaudio, av, numpy.
- Só copia se o arquivo já for WAV com sample_rate e canais iguais ao alvo
  E for PCM16. Caso contrário, decodifica e normaliza.

Heurística de decisão:
- Extensão é pista inicial.
- Sniff com PyAV valida conteúdo quando for útil (extensões de áudio) para detectar contêiner/vídeo.
- Se torchaudio falhar ao carregar, fallback para PyAV decode.
"""
from pathlib import Path
import shutil

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import av


# Formatos de áudio comuns que o torchaudio costuma carregar bem.
AUDIO_EXTS_TORCHAUDIO = {
    "wav",
    "flac",
    "ogg",
    "oga",
    "mp3",
    "wma",
}

# Contêineres: delegar ao PyAV para demux/decodificar.
CONTAINER_EXTS_PYAV = {
    "mp4",
    "mkv",
    "mov",
    "avi",
    "webm",
    "m4a",
}


class AudioExtractionError(RuntimeError):
    pass


class AudioExtractor:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype=torch.float32):
        """
        sample_rate: target sample rate (default 16000)
        channels: target number of channels (default 1 = mono)
        dtype: torch dtype used during processing (default float32)
        """
        self.target_sr = int(sample_rate)
        self.target_channels = int(channels)
        self.dtype = dtype

    # ---------------------------
    # Inspeção e decisão
    # ---------------------------
    def _inspect_wav_info(self, inp: Path) -> dict:
        """
        Inspeciona metadados de um WAV via torchaudio.info.
        Retorna dict com sample_rate, channels, bits_per_sample, num_frames, encoding.
        """
        info = torchaudio.info(str(inp))
        return {
            "sample_rate": int(info.sample_rate) if info.sample_rate is not None else None,
            "channels": int(getattr(info, "num_channels", None)) if getattr(info, "num_channels", None) is not None else None,
            "bits_per_sample": int(getattr(info, "bits_per_sample", -1)) if getattr(info, "bits_per_sample", None) is not None else None,
            "num_frames": int(getattr(info, "num_frames", -1)) if getattr(info, "num_frames", None) is not None else None,
            "encoding": getattr(info, "encoding", None) or getattr(info, "format", None),
        }

    def _copy_if_wav_already_ok(self, inp: Path, out: Path) -> bool:
        """
        Se o arquivo for WAV e já estiver exatamente no alvo (PCM16, sample_rate e canais),
        copia e retorna True. Caso contrário, retorna False.
        """
        ext = inp.suffix.lower().lstrip(".")
        if ext != "wav":
            return False

        meta = self._inspect_wav_info(inp)
        sr = meta.get("sample_rate")
        ch = meta.get("channels")
        bps = meta.get("bits_per_sample")
        enc = (meta.get("encoding") or "").upper()

        if sr == self.target_sr and ch == self.target_channels:
            # Verificação de PCM16 (preferencial). Se disponível, exija 16 bits ou encoding PCM_S.
            is_pcm16 = (bps == 16) or ("PCM_S" in enc or "PCM" in enc)
            if is_pcm16:
                out.parent.mkdir(parents=True, exist_ok=True)
                if inp.resolve() != out.resolve():
                    shutil.copy2(str(inp), str(out))
                print(f"[extract_audio] WAV já está no formato alvo (PCM16, {self.target_sr} Hz, {self.target_channels} ch) — cópia realizada.")
                return True

        return False

    def _sniff_with_pyav(self, inp: Path) -> dict:
        """
        'Fareja' o conteúdo com PyAV para decidir melhor o backend.
        Retorna dict com ok, has_audio, has_video, codec, sample_rate, channels.
        Não lança exceção; em caso de falha retorna ok=False.
        """
        try:
            container = av.open(str(inp))
            a_streams = list(container.streams.audio)
            v_streams = list(container.streams.video)
            has_audio = len(a_streams) > 0
            has_video = len(v_streams) > 0
            codec_name = None
            sr = None
            ch = None
            if has_audio:
                cctx = a_streams[0].codec_context
                codec_name = getattr(cctx, "name", None)
                sr = int(cctx.sample_rate) if cctx and cctx.sample_rate else None
                ch = int(cctx.channels) if cctx and cctx.channels else None
            return {
                "ok": True,
                "has_audio": has_audio,
                "has_video": has_video,
                "codec": codec_name,
                "sample_rate": sr,
                "channels": ch,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------------------
    # Leitura/decodificação
    # ---------------------------
    def _load_with_torchaudio(self, inp: Path) -> tuple[torch.Tensor, int]:
        """
        Carrega áudio com torchaudio.load.
        Retorna (tensor [channels, samples], sample_rate).
        """
        wav, sr = torchaudio.load(str(inp))  # float32 [-1, 1] por padrão
        return wav.to(self.dtype), int(sr)

    def _decode_with_pyav(self, inp: Path) -> tuple[torch.Tensor, int]:
        """
        Decodifica áudio de contêiner com PyAV.
        Usa o primeiro stream de áudio (decode(audio=0)) e trata variações de to_ndarray across PyAV versions.
        Retorna (tensor [channels, samples], sample_rate) com dtype float32.
        """
        container = av.open(str(inp))
        audio_streams = list(container.streams.audio)
        if not audio_streams:
            raise AudioExtractionError("Nenhum stream de áudio encontrado no arquivo (PyAV).")

        # Primeiro stream de áudio
        audio_stream = audio_streams[0]
        sample_rate = int(audio_stream.codec_context.sample_rate)

        frames = []
        for frame in container.decode(audio=0):  # posição 0 dentro dos streams de áudio
            # Use frame.to_ndarray() sem argumentos — versões diferentes do PyAV podem aceitar 'format' ou não.
            arr = frame.to_ndarray()
            if arr is None:
                continue
            arr = np.asarray(arr)

            # Normalizar para (channels, samples)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim == 2:
                codec_ch = getattr(audio_stream.codec_context, "channels", None)
                if codec_ch is not None:
                    if arr.shape[0] == codec_ch:
                        pass
                    elif arr.shape[1] == codec_ch:
                        arr = arr.T
                    else:
                        if arr.shape[0] < arr.shape[1]:
                            pass
                        else:
                            arr = arr.T
                else:
                    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                        pass
                    else:
                        arr = arr.T
            else:
                arr = arr.reshape(arr.shape[0], -1)

            if arr.size == 0:
                continue

            # Normalize dtype to float32 in -1..1
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                max_abs = max(abs(info.min), abs(info.max))
                arr = arr.astype(np.float32) / float(max_abs)
            else:
                arr = arr.astype(np.float32)

            frames.append(arr)

        if not frames:
            raise AudioExtractionError("PyAV não decodificou frames de áudio.")

        audio = np.concatenate(frames, axis=1)  # channels x samples
        tensor = torch.from_numpy(audio).to(self.dtype)
        return tensor, sample_rate

    # ---------------------------
    # Normalização e salvamento
    # ---------------------------
    def _to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Garante [channels, samples]; se >1 canais, média para mono.
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def _resample_if_needed(self, wav: torch.Tensor, src_sr: int) -> tuple[torch.Tensor, int]:
        """
        Aplica resample para target_sr se necessário.
        """
        if src_sr == self.target_sr:
            return wav, src_sr
        if wav.dtype != self.dtype:
            wav = wav.to(self.dtype)
        resampler = T.Resample(orig_freq=src_sr, new_freq=self.target_sr, dtype=self.dtype)
        wav_rs = resampler(wav)
        return wav_rs, self.target_sr

    def _match_channels(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Ajusta o número de canais conforme self.target_channels.
        - Se alvo = 1: converte para mono.
        - Se alvo > 1: replica/trunca canais conforme necessário.
        """
        if self.target_channels == 1:
            return self._to_mono(wav)

        # Garantir [C, T]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        c = wav.size(0)
        if c == self.target_channels:
            return wav
        if c > self.target_channels:
            return wav[: self.target_channels, :]
        # c < target: replicar último canal
        rep = self.target_channels - c
        return torch.cat([wav, wav[-1:].repeat(rep, 1)], dim=0)

    def _save_wav_pcm16(self, wav: torch.Tensor, sample_rate: int, out: Path):
        """
        Salva WAV PCM16 sem depender de torchaudio.save (usa módulo wave da stdlib).
        Espera wav: torch.Tensor [channels, samples] float em [-1, 1].
        """
        import wave

        out.parent.mkdir(parents=True, exist_ok=True)

        # Garantir tensor float32 na CPU
        if isinstance(wav, torch.Tensor):
            wav_cpu = wav.detach().cpu().to(torch.float32)
            arr = wav_cpu.numpy()
        else:
            arr = np.array(wav, dtype=np.float32)

        # Garantir shape (channels, samples)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        # Clip em [-1, 1]
        arr = np.clip(arr, -1.0, 1.0)

        # Converter para int16 PCM
        int16 = (arr * 32767.0).astype(np.int16)

        # Transpor para (samples, channels)
        frames = int16.T  # shape (samples, channels)

        n_channels = frames.shape[1]
        sampwidth = 2  # bytes por sample (16-bit)
        n_frames = frames.shape[0]

        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(int(n_channels))
            wf.setsampwidth(int(sampwidth))
            wf.setframerate(int(sample_rate))
            wf.writeframes(frames.tobytes())

    # ---------------------------
    # Pipeline principal
    # ---------------------------
    def extract(self, input_path: str, output_wav: str, overwrite: bool = True) -> Path:
        """
        Extrai e normaliza áudio de input_path, salvando em output_wav.
        Decisão:
        - Se for WAV e já estiver em (PCM16, target_sr, target_channels) -> copia.
        - Se extensão em AUDIO_EXTS_TORCHAUDIO -> usa torchaudio.load (sniff com PyAV para detectar vídeo).
        - Caso contrário -> usa PyAV.
        - Fallback: se torchaudio falhar ao carregar, decoder com PyAV.
        """
        inp = Path(input_path)
        out = Path(output_wav)

        if not inp.exists():
            raise AudioExtractionError(f"Arquivo de entrada não encontrado: {inp}")

        if out.exists() and not overwrite:
            print(f"[extract_audio] Saída já existe e overwrite=False — retornando: {out}")
            return out

        ext = inp.suffix.lower().lstrip(".")

        # 1) Se WAV já perfeito, copiar e retornar.
        if ext == "wav" and self._copy_if_wav_already_ok(inp, out):
            return out

        # 2) Seleção de backend com sniff (quando útil)
        used_backend = None
        if ext in AUDIO_EXTS_TORCHAUDIO:
            sniff = self._sniff_with_pyav(inp)
            if sniff.get("ok") and sniff.get("has_video"):
                # Extensão "de áudio", mas conteúdo tem vídeo -> use PyAV
                wav, sr = self._decode_with_pyav(inp)
                used_backend = "pyav"
            else:
                # Tente torchaudio primeiro, com fallback para PyAV se falhar
                try:
                    wav, sr = self._load_with_torchaudio(inp)
                    used_backend = "torchaudio"
                except Exception:
                    wav, sr = self._decode_with_pyav(inp)
                    used_backend = "pyav"
        else:
            # Contêineres / extensões desconhecidas -> PyAV
            wav, sr = self._decode_with_pyav(inp)
            used_backend = "pyav"

        # 3) Ajuste de canais e resample
        wav = self._match_channels(wav)
        wav, final_sr = self._resample_if_needed(wav, sr)

        # 4) Salvar WAV PCM16 (sem torchaudio.save)
        self._save_wav_pcm16(wav, final_sr, out)

        # 5) Log
        duration = wav.shape[-1] / final_sr if final_sr > 0 else None
        print(f"[extract_audio] OK backend={used_backend} sr_in={sr} -> sr_out={final_sr} channels_out={wav.shape[0]} duration_s={duration:.2f}")

        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrai/normaliza áudio (torchaudio preferencial; PyAV para contêineres; sniff de conteúdo quando útil).")
    parser.add_argument("input", help="arquivo de entrada (mp4/mkv/mov/avi/webm/m4a/wav/flac/ogg/mp3/...)")
    parser.add_argument("output", help="arquivo WAV de saída")
    parser.add_argument("--sample-rate", type=int, default=16000, help="sample rate alvo (padrão 16000)")
    parser.add_argument("--channels", type=int, default=1, help="número de canais alvo (padrão 1 = mono)")
    parser.add_argument("--no-overwrite", action="store_true", help="não sobrescrever saída existente")
    args = parser.parse_args()

    extractor = AudioExtractor(sample_rate=args.sample_rate, channels=args.channels)
    out_path = extractor.extract(args.input, args.output, overwrite=not args.no_overwrite)
    print("Arquivo de áudio gerado:", out_path)