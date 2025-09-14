#!/usr/bin/env python3
# coding: utf-8
"""
utils/ollama_client.py

Cliente para Ollama com:
- Geração (prompt + 0..N imagens)
- Inventário e instalação de modelos (list, show, pull)
- Detecção de capacidades (suporta imagens? suporta múltiplas imagens?)
- Recomendações por tarefa (geral/visão, OCR, multi-imagem)
- safe_generate: decide automaticamente multi x single imagem com fallback

Observações importantes:
- A API do Ollama evolui; os payloads de visão podem variar entre modelos/versões.
- Por padrão, este cliente envia imagens como data URI ("data:image/jpeg;base64,...") em payload["images"].
  Se precisar, você pode enviar como base64 puro (images_as_data_uris=False).
- A detecção de capacidades usa heurística por nome + (opcional) probe ativo (requisições rápidas de teste).
  Ative probe quando quiser confirmação automática (use_probe=True).

Dependências:
- requests (obrigatório)
- pillow (opcional, usado para converter numpy/PIL em JPEG e para probe ativo)
- numpy (opcional, se você passar arrays)

Uso rápido:
  client = OllamaClient("http://localhost:11434")
  client.ensure_model("gemma3:12b")   # baixa se faltar
  caps = {
    "gemma3:12b": {
      "vision": client.supports_images("gemma3:12b"),
      "multi": client.supports_multi_images("gemma3:12b"),
    }
  }
  res = client.safe_generate("gemma3:12b", "Descreva o contexto", images=[np_arr1, np_arr2])
  print(res["text"])
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import requests

# imports opcionais em runtime
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # fallback

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # fallback


ImageLike = Union["np.ndarray", "bytes", "PIL.Image.Image"]  # para hints


class OllamaClient:
    """
    Cliente mínimo e extensível para o Ollama local.

    Parâmetros:
      host: URL do servidor Ollama (ex.: "http://localhost:11434")
      session: optional requests.Session() para reuso; se None, cria internamente
      max_retries: tentativas em falha de rede/transiente em chamadas simples (generate, show, etc.)
      backoff: tempo base para backoff exponencial entre tentativas (segundos)
    """

    def __init__(self, host: str = "http://localhost:11434", session: Optional[requests.Session] = None, max_retries: int = 2, backoff: float = 0.5):
        self.host = host.rstrip("/")
        self.session = session or requests.Session()
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        # cache leve de capacidades: {model_name: {"vision": bool|None, "multi": bool|None, "probed": bool}}
        self._cap_cache: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Helpers de imagem e base64
    # -------------------------
    def _pil_from_numpy(self, arr: "np.ndarray") -> "Image.Image":
        if Image is None:
            raise RuntimeError("Pillow não está disponível para converter numpy -> imagem. Instale 'pillow'.")
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "RGB"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError("Formato de numpy array não suportado para conversão em imagem.")
        return Image.fromarray(arr.astype("uint8"), mode=mode)

    def _image_to_jpeg_bytes(self, im: ImageLike, quality: int = 90) -> bytes:
        """
        Converte PIL.Image, numpy.ndarray ou bytes (assume já imagem codificada) para bytes JPEG.
        - Se bytes forem passados, retornamos como estão (assumimos já codificados em algum formato).
        """
        if isinstance(im, (bytes, bytearray)):
            return bytes(im)
        if np is not None and isinstance(im, np.ndarray):
            pil = self._pil_from_numpy(im)
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        if Image is not None and isinstance(im, Image.Image):
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        raise ValueError("Tipo de imagem não suportado. Forneça numpy.ndarray, PIL.Image.Image ou bytes.")

    def _to_base64(self, data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    def _to_data_uri_base64(self, img_bytes: bytes, mime: str = "image/jpeg") -> str:
        return f"data:{mime};base64,{self._to_base64(img_bytes)}"

    def _make_probe_image_bytes(self) -> Optional[bytes]:
        """
        Cria uma imagem minúscula para probe (branca 4x4). Retorna bytes JPEG ou None se não possível.
        """
        if Image is None:
            return None
        img = Image.new("RGB", (4, 4), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    # -------------------------
    # Helpers de resposta
    # -------------------------
    @staticmethod
    def _extract_text_from_response_json(resp_json: Dict[str, Any]) -> str:
        if not isinstance(resp_json, dict):
            try:
                return str(resp_json)
            except Exception:
                return ""
        if "text" in resp_json and isinstance(resp_json["text"], str):
            return resp_json["text"]
        out = resp_json.get("output")
        if isinstance(out, str):
            return out
        if isinstance(out, list):
            try:
                return " ".join([str(x) for x in out])
            except Exception:
                pass
        results = resp_json.get("results")
        if isinstance(results, list) and results:
            parts = []
            for r in results:
                if isinstance(r, dict):
                    for k in ("content", "text", "output"):
                        v = r.get(k)
                        if isinstance(v, str):
                            parts.append(v)
            if parts:
                return "\n".join(parts)
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            parts = []
            for ch in choices:
                if isinstance(ch, dict):
                    t = ch.get("text") or ch.get("message") or ch.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "\n".join(parts)
        try:
            return json.dumps(resp_json, ensure_ascii=False)
        except Exception:
            return str(resp_json)

    # -------------------------
    # Inventário / versão / show
    # -------------------------
    def get_version(self, timeout: int = 10) -> Dict[str, Any]:
        """
        GET /api/version — retorna versão do servidor (se suportado).
        Alguns servidores podem não expor; neste caso retorna {}.
        """
        url = f"{self.host}/api/version"
        try:
            r = self.session.get(url, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"version": r.text.strip()}
            else:
                return {}
        except Exception:
            return {}

    def list_models(self, timeout: int = 30) -> List[str]:
        """
        GET /api/tags — retorna nomes dos modelos instalados.
        """
        url = f"{self.host}/api/tags"
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                models.append(name)
        return models

    def show_model(self, name: str, timeout: int = 30) -> Dict[str, Any]:
        """
        POST /api/show — retorna metadados do modelo.
        """
        url = f"{self.host}/api/show"
        r = self.session.post(url, json={"name": name}, timeout=timeout)
        if r.status_code != 200:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise RuntimeError(f"Falha ao consultar modelo '{name}': {detail}")
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}

    def is_installed(self, name: str, timeout: int = 30) -> bool:
        """
        Verifica se o modelo 'name' aparece em /api/tags.
        """
        try:
            models = self.list_models(timeout=timeout)
            return any(m.lower() == name.lower() for m in models)
        except Exception:
            return False

    def ensure_model(
        self,
        name: str,
        pull_if_missing: bool = True,
        stream_progress: bool = True,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        timeout: int = 600,
    ) -> bool:
        """
        Garante que o modelo esteja instalado. Se não estiver e pull_if_missing=True, executa /api/pull.

        Retorna True se o modelo estiver instalado ao final (já estava ou foi baixado).
        """
        if self.is_installed(name):
            return True
        if not pull_if_missing:
            return False

        url = f"{self.host}/api/pull"
        try:
            if stream_progress:
                # Streaming de progresso linha a linha (cada linha é um JSON)
                with self.session.post(url, json={"name": name}, timeout=timeout, stream=True) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except Exception:
                            msg = {"status": line}
                        if on_progress:
                            try:
                                on_progress(msg)
                            except Exception:
                                pass
                        # Alguns servidores sinalizam conclusão com {"status":"success"} ou "download complete"
                        status = str(msg.get("status", "")).lower()
                        if "error" in msg:
                            raise RuntimeError(f"Erro no pull de '{name}': {msg['error']}")
                        # Não encerramos aqui; deixamos o stream concluir naturalmente
                # Após concluir, validamos
                return self.is_installed(name)
            else:
                # Sem streaming: faz POST e espera terminar
                resp = self.session.post(url, json={"name": name}, timeout=timeout)
                resp.raise_for_status()
                # Algumas versões retornam um JSON final
                return self.is_installed(name)
        except Exception as e:
            raise RuntimeError(f"Falha ao baixar/instalar modelo '{name}': {e}")

    # -------------------------
    # Heurísticas de capacidades por nome
    # -------------------------
    @staticmethod
    def _guess_caps_by_name(name: str) -> Dict[str, bool]:
        """
        Heurística por nome do modelo para capacidades:
        - vision: aceita imagens
        - multi: aceita múltiplas imagens por requisição
        - ocr: tende a ir bem em OCR
        """
        nm = name.lower()
        caps = {"vision": False, "multi": False, "ocr": False}

        patterns_vision_multi = [
            r"gemma3",            # ex: gemma3:12b (tende a aceitar multi-imagem)
            r"qwen.*(vl|vision)", # ex: qwen2-vl, qwen2.5-vl
            r"llama.*vision",     # ex: llama3.2-vision
        ]
        patterns_vision_single = [
            r"minicpm[-_]?v",     # MiniCPM-V — visão forte/OCR, tipicamente single-image
            r"llava",             # LLaVA / bakllava — podem ser single-image em algumas builds
            r"bakllava",
            r"blip",              # BLIP variantes (se existir no seu catálogo local)
        ]
        patterns_ocr = [
            r"minicpm[-_]?v",
            r"qwen.*(vl|vision)",
            r"paddleocr",  # se existir uma ponte
        ]

        if any(re.search(p, nm) for p in patterns_vision_multi):
            caps["vision"] = True
            caps["multi"] = True
        if any(re.search(p, nm) for p in patterns_vision_single):
            caps["vision"] = True
            caps["multi"] = caps["multi"] or False
        if any(re.search(p, nm) for p in patterns_ocr):
            caps["ocr"] = True

        return caps

    # -------------------------
    # Detecção ativa (probe) de capacidades
    # -------------------------
    def supports_images(self, name: str, use_probe: bool = False, timeout: int = 20) -> bool:
        """
        Retorna True se o modelo parece suportar imagens.
        - Heurística por nome (sempre aplicada)
        - Probe ativo: tenta enviar uma imagem 4x4 branca; se retornar 200, consideramos suportado.
        """
        cache = self._cap_cache.get(name, {})
        if "vision" in cache and cache["vision"] is not None and not use_probe:
            return bool(cache["vision"])

        caps = self._guess_caps_by_name(name)
        vision = caps["vision"]

        if use_probe:
            probe_img = self._make_probe_image_bytes()
            if probe_img is not None:
                try:
                    _ = self.generate(
                        model=name,
                        prompt="Teste rápido: descreva a imagem.",
                        images=[probe_img],
                        timeout=timeout,
                        # limitar opções para não gastar tokens/dinheiro, se aplicável
                        options={"max_tokens": 8},
                    )
                    vision = True
                except Exception:
                    # Se falhar no generate com imagem, assume não suportado
                    vision = False

        self._cap_cache.setdefault(name, {}).update({"vision": vision, "probed": self._cap_cache.get(name, {}).get("probed", False) or use_probe})
        return bool(vision)

    def supports_multi_images(self, name: str, use_probe: bool = False, timeout: int = 20) -> bool:
        """
        Retorna True se o modelo parece aceitar múltiplas imagens na mesma requisição.
        - Heurística por nome (gemma3, qwen-vl, llama-vision → True; minicpm-v, llava → False)
        - Probe ativo (opcional): tenta enviar 2 imagens no mesmo /api/generate.
        """
        cache = self._cap_cache.get(name, {})
        if "multi" in cache and cache["multi"] is not None and not use_probe:
            return bool(cache["multi"])

        caps = self._guess_caps_by_name(name)
        multi = caps["multi"]

        if use_probe and self.supports_images(name, use_probe=False):
            probe_img = self._make_probe_image_bytes()
            if probe_img is not None:
                try:
                    _ = self.generate(
                        model=name,
                        prompt="Teste multi-imagem (2x).",
                        images=[probe_img, probe_img],
                        timeout=timeout,
                        options={"max_tokens": 8},
                    )
                    multi = True
                except Exception:
                    multi = False

        self._cap_cache.setdefault(name, {}).update({"multi": multi, "probed": self._cap_cache.get(name, {}).get("probed", False) or use_probe})
        return bool(multi)

    # -------------------------
    # Recomendações de modelos por tarefa
    # -------------------------
    def recommend_models(self) -> Dict[str, List[str]]:
        """
        Lista de recomendações (ordem de preferência) por tarefa.
        Ajuste conforme seu catálogo local.
        """
        return {
            # Contexto geral de imagens e multi-imagem
            "general": ["gemma3:12b", "qwen2-vl", "llama3.2-vision:11b", "llava"],
            # Tarefas de OCR
            "ocr": ["minicpm-v", "qwen2-vl", "llava"],
            # Explicitamente multi-imagem
            "multi": ["gemma3:12b", "qwen2-vl", "llama3.2-vision:11b"],
        }

    def ensure_recommended(self, task: str, pull_if_missing: bool = True) -> List[str]:
        """
        Garante que pelo menos um modelo recomendado para 'task' esteja disponível (instalado).
        Retorna a lista de modelos recomendados que já estão instalados (após tentar instalar o primeiro, se permitido).
        """
        recs = self.recommend_models().get(task, [])
        installed = []
        for i, name in enumerate(recs):
            if self.is_installed(name):
                installed.append(name)
                continue
            if i == 0 and pull_if_missing:
                ok = self.ensure_model(name, pull_if_missing=True)
                if ok:
                    installed.append(name)
        # Reavalia os demais (sem baixar)
        for name in recs:
            if name not in installed and self.is_installed(name):
                installed.append(name)
        return installed

    # -------------------------
    # Método principal: generate
    # -------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[Iterable[ImageLike]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        max_images_per_request: Optional[int] = None,
        images_as_data_uris: bool = True,
        image_mime: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """
        Gera uma resposta do modelo Ollama, enviando prompt e opcionalmente imagens.

        - images_as_data_uris=True: envia cada imagem como "data:<mime>;base64,<...>" (compatível com diversos modelos).
          Se False: envia lista de strings base64 puras (alguns modelos/servidores preferem assim).
        - image_mime: mime usado nos data URIs (se aplicável).
        """
        # Prepara imagens -> lista de representações (data URI ou base64 puro)
        img_payload: List[Union[str, Dict[str, str]]] = []
        if images is not None:
            for im in images:
                try:
                    b = self._image_to_jpeg_bytes(im)
                    if images_as_data_uris:
                        img_payload.append(self._to_data_uri_base64(b, mime=image_mime))
                    else:
                        img_payload.append(self._to_base64(b))
                except Exception as e:
                    raise RuntimeError(f"Erro convertendo imagem para bytes: {e}")

        # Batching básico
        if not img_payload:
            batches = [None]
        else:
            if max_images_per_request is None or max_images_per_request <= 0:
                batches = [img_payload]
            else:
                batches = [img_payload[i : i + max_images_per_request] for i in range(0, len(img_payload), max_images_per_request)]

        combined_texts: List[str] = []
        raw_responses: List[Dict[str, Any]] = []
        last_status = None

        url = f"{self.host}/api/generate"
        for batch in batches:
            payload: Dict[str, Any] = {"model": model, "prompt": prompt}
            if batch is not None:
                # Alguns servidores aceitam "images": [data_uri | base64, ...]
                payload["images"] = batch
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if options:
                payload.update(options)

            # retries simples
            attempt = 0
            while True:
                attempt += 1
                try:
                    resp = self.session.post(url, json=payload, timeout=timeout)
                    last_status = resp.status_code
                    if resp.status_code != 200:
                        try:
                            err = resp.json()
                        except Exception:
                            err = resp.text
                        raise RuntimeError(f"Ollama retornou status {resp.status_code}: {err}")
                    try:
                        resp_json = resp.json()
                    except Exception:
                        resp_json = {"text": resp.text}
                    txt = self._extract_text_from_response_json(resp_json)
                    combined_texts.append(txt)
                    raw_responses.append(resp_json)
                    break
                except Exception as e:
                    if attempt > self.max_retries:
                        raise RuntimeError(f"Falha ao chamar Ollama ({url}) após {attempt} tentativas: {e}")
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
                    continue

        final_text = "\n\n".join([t for t in combined_texts if t])
        result = {
            "text": final_text,
            "raw": raw_responses[0] if len(raw_responses) == 1 else raw_responses,
            "status_code": last_status,
            "model": model,
        }
        return result

    # -------------------------
    # safe_generate: adapta a estratégia conforme capacidades
    # -------------------------
    def safe_generate(
        self,
        model: str,
        prompt: str,
        images: Optional[Iterable[ImageLike]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        ensure_if_missing: bool = True,
        use_probe: bool = False,
        prefer_multi: bool = True,
        images_as_data_uris: bool = True,
    ) -> Dict[str, Any]:
        """
        Geração segura com fallback automático:
        - Se não houver imagens: chama generate normal.
        - Se houver imagens:
          - Verifica se o modelo suporta visão (heurística + opcional probe). Se não, lança erro informativo.
          - Se prefer_multi=True e modelo suporta multi-imagem: envia tudo numa única chamada.
          - Caso contrário, faz N chamadas (uma por imagem) e concatena as respostas.

        ensure_if_missing: baixa o modelo se não estiver instalado.
        use_probe: faz probes ativos (recomendo ligar ao menos uma vez por sessão se estiver em dúvida).
        """
        if ensure_if_missing and not self.is_installed(model):
            self.ensure_model(model, pull_if_missing=True)

        imgs = list(images) if images is not None else []

        if not imgs:
            return self.generate(
                model=model,
                prompt=prompt,
                images=None,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
                timeout=timeout,
                images_as_data_uris=images_as_data_uris,
            )

        # Com imagens
        if not self.supports_images(model, use_probe=use_probe):
            recs = self.recommend_models()
            suggestion = recs.get("general", [])[:2]
            raise RuntimeError(f"O modelo '{model}' não parece suportar imagens. Sugestões: {suggestion}")

        multi = self.supports_multi_images(model, use_probe=use_probe)
        if prefer_multi and multi:
            # uma chamada só com todas as imagens
            return self.generate(
                model=model,
                prompt=prompt,
                images=imgs,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
                timeout=timeout,
                images_as_data_uris=images_as_data_uris,
            )
        else:
            # fallback: uma chamada por imagem, concatena
            parts: List[str] = []
            raws: List[Any] = []
            last_status = None
            for i, im in enumerate(imgs, start=1):
                pfx = f"[imagem {i}/{len(imgs)}]\n"
                res = self.generate(
                    model=model,
                    prompt=prompt,
                    images=[im],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    options=options,
                    timeout=timeout,
                    images_as_data_uris=images_as_data_uris,
                )
                parts.append(pfx + (res.get("text") or ""))
                raws.append(res.get("raw"))
                last_status = res.get("status_code")
            return {
                "text": "\n\n".join(parts),
                "raw": raws,
                "status_code": last_status,
                "model": model,
            }


__all__ = ["OllamaClient"]