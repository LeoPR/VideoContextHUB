#!/usr/bin/env python3
# coding: utf-8
"""
utils/audio_io.py

API centralizada para leitura/gravação de áudio do projeto.

- Backend principal: soundfile (pysoundfile/libsndfile).
- Retorna numpy.float32 1-D waveforms mono pronto para uso com Resemblyzer.
- Permite leitura inteira (.load()) ou janelas (.windows(window_s, hop_s)) sem
  carregar tudo na RAM de forma explícita (usa seek+read por janela).
- AudioWriter para escrever WAV com controle de samplerate/channels/subtype.

Uso:
  from utils.audio_io import AudioReader, load_wav_as_numpy

  with AudioReader("path/to/file.wav") as r:
      info = r.info()
      wav, sr = r.load()
      for start_s, end_s, samples in r.windows(1.0, 0.5):
          # process samples (numpy float32 mono)
"""
from __future__ import annotations
from typing import Generator, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import soundfile as sf
import contextlib


class AudioReader:
    """
    Context manager para leitura de arquivos de áudio.

    Exemplos:
      with AudioReader("audio.wav") as reader:
          info = reader.info()
          wav, sr = reader.load()
          for s, e, samples in reader.windows(1.0, 0.5):
              ...

    Observações:
    - Sempre retorna numpy float32 mono 1-D (se input for multicanal, faz média dos canais).
    - A leitura por janelas usa seek+read para evitar alocar tudo de uma vez (útil para arquivos longos).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._sf: Optional[sf.SoundFile] = None
        self.samplerate: Optional[int] = None
        self.channels: Optional[int] = None
        self.frames: Optional[int] = None

    def __enter__(self) -> "AudioReader":
        self._sf = sf.SoundFile(str(self.path), mode="r")
        self.samplerate = self._sf.samplerate
        self.channels = self._sf.channels
        self.frames = len(self._sf)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
        self._sf = None

    def info(self) -> Dict:
        """
        Retorna metadados do arquivo: samplerate, channels, frames, duration, format, subtype.
        """
        if self._sf is None:
            with self:
                return self.info()
        return {
            "samplerate": int(self.samplerate),
            "channels": int(self.channels),
            "frames": int(self.frames),
            "duration": float(self.frames / self.samplerate) if self.samplerate else 0.0,
            "format": getattr(self._sf, "format", None),
            "subtype": getattr(self._sf, "subtype", None),
        }

    def load(self) -> Tuple[np.ndarray, int]:
        """
        Carrega o arquivo inteiro e retorna (waveform_1d_float32, samplerate).
        Se o áudio for multicanal, retorna média dos canais (mono).
        """
        if self._sf is None:
            with self:
                return self.load()
        # seek to start
        self._sf.seek(0)
        data = self._sf.read(dtype="float32", always_2d=False)
        if data is None:
            return np.zeros(0, dtype=np.float32), int(self.samplerate)
        if self.channels and self.channels > 1:
            # média dos canais -> mono
            data = np.mean(data, axis=1)
        return data.astype(np.float32, copy=False), int(self.samplerate)

    def read_frames(self, start_frame: int, num_frames: int) -> np.ndarray:
        """
        Lê num_frames a partir de start_frame (índices em samples).
        Retorna numpy float32 1-D (mono).
        """
        if self._sf is None:
            with self:
                return self.read_frames(start_frame, num_frames)
        # Clamp
        start_frame = max(0, int(start_frame))
        if start_frame >= self.frames:
            return np.zeros(0, dtype=np.float32)
        self._sf.seek(start_frame)
        to_read = int(min(num_frames, self.frames - start_frame))
        data = self._sf.read(frames=to_read, dtype="float32", always_2d=False)
        if data is None:
            return np.zeros(0, dtype=np.float32)
        if self.channels and self.channels > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32, copy=False)

    def windows(self, window_size_s: float, hop_s: float) -> Generator[Tuple[float, float, np.ndarray], None, None]:
        """
        Generator que itera sobre janelas (start_s, end_s, samples) do arquivo.
        Cada samples é numpy float32 1-D.

        Exemplo:
          for start, end, samples in reader.windows(1.0, 0.5):
              ...

        Observação: a leitura é feita por seek+read; eficiente para janelamento sem carregar tudo.
        """
        if self._sf is None:
            with self:
                yield from self.windows(window_size_s, hop_s)
                return

        sr = int(self.samplerate)
        win_frames = max(1, int(round(window_size_s * sr)))
        hop_frames = max(1, int(round(hop_s * sr)))

        start_frame = 0
        while start_frame < self.frames:
            end_frame = min(self.frames, start_frame + win_frames)
            samples = self.read_frames(start_frame, end_frame - start_frame)
            start_s = float(start_frame / sr)
            end_s = float(end_frame / sr)
            yield (start_s, end_s, samples)
            start_frame += hop_frames


class AudioWriter:
    """
    Simple writer wrapper around soundfile.SoundFile for gravação.
    """

    def __init__(self, path: str | Path, samplerate: int, channels: int = 1, subtype: str = "PCM_16", format: str = "WAV"):
        self.path = Path(path)
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.subtype = subtype
        self.format = format
        self._sf: Optional[sf.SoundFile] = None

    def __enter__(self) -> "AudioWriter":
        self._sf = sf.SoundFile(str(self.path), mode="w", samplerate=self.samplerate, channels=self.channels, subtype=self.subtype, format=self.format)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
        self._sf = None

    def write(self, samples: np.ndarray):
        """
        Escreve samples no arquivo aberto. Aceita 1D (mono) ou 2D (frames, channels).
        """
        if self._sf is None:
            raise RuntimeError("AudioWriter não está aberto. Use with AudioWriter(...) as w:")
        arr = np.asarray(samples, dtype="float32")
        if arr.ndim == 1:
            if self.channels != 1:
                # replicate mono to channels
                arr = np.tile(arr[:, None], (1, self.channels))
        elif arr.ndim == 2:
            if arr.shape[1] != self.channels:
                raise ValueError(f"Samples têm {arr.shape[1]} canais, configurado writer espera {self.channels}")
        else:
            raise ValueError("Samples deve ser 1D ou 2D ndarray.")
        self._sf.write(arr)

    def close(self):
        if self._sf is not None:
            try:
                self._sf.close()
            except Exception:
                pass
            self._sf = None


# Helper de conveniência
def load_wav_as_numpy(path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Conveniência: carrega inteiro e retorna (waveform_1d_float32_mono, samplerate).
    """
    with AudioReader(path) as reader:
        return reader.load()