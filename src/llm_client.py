"""Cliente mínimo para la API local de Ollama."""
from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434"
MODELO_POR_DEFECTO = "llama3.2:3b"


class OllamaClient:
    def __init__(self, modelo: str = MODELO_POR_DEFECTO, base_url: str = OLLAMA_URL):
        self.modelo = modelo
        self.base_url = base_url.rstrip("/")

    def estado(self) -> tuple[bool, list[str], str]:
        """(conectado, modelos instalados, mensaje)."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            modelos = [m.get("name", "") for m in r.json().get("models", [])]
            return True, modelos, "Ollama conectado"
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            return False, [], f"Ollama no disponible: {exc}"

    def chat(self, sistema: str, usuario: str, temperatura: float = 0.3, max_tokens: int = 450) -> str:
        payload = {
            "model": self.modelo,
            "stream": False,
            "messages": [
                {"role": "system", "content": sistema},
                {"role": "user", "content": usuario},
            ],
            "options": {"temperature": temperatura, "num_predict": max_tokens},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=240)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
