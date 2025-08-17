"""
Lightweight Ollama client.
    - /api/chat + /api/generate fallback
    - Windows support: specify absolute path to ollama.exe if not in PATH
"""

import requests
import json
import subprocess
import os

# Constants.
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"
GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"

# NOTE: CHANGE THIS FOR WINDOWS!
# Use "ollama" if callable from PATH.
# Otherwise, set the full .exe path here.
OLLAMA_BIN = r"C:\Users\theac\AppData\Local\Programs\Ollama\ollama app.exe"  # Set full .exe path here if needed.


def _autopull_model(model: str):
    """
    Call 'ollama pull <model>' if <model> has not yet been pulled.
    """
    print(f"[ollama-client] Pulling missing model '{model}'...")
    try:
        subprocess.run([OLLAMA_BIN, "pull", model], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not run {OLLAMA_BIN}. "
            f"Update OLLAMA_BIN in llm_client.py to the full path of your 'ollama.exe'."
        )
    print(f"processor: [ollama-client] Model '{model}' pulled.")


def ollama_generate(prompt: str, model: str = "llama3", max_tokens: int = 2048) -> str:
    """
    Interface with ollama API.
    """

    tried_autopull = False

    while True:
        # Attempt to /api/chat.
        try:
            chat_payload = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
            r = requests.post(CHAT_ENDPOINT, data=json.dumps(chat_payload), headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                return r.json()["message"]["content"].strip()
            if (r.status_code == 404) and (not tried_autopull):
                tried_autopull = True
                _autopull_model(model)
                continue
        except requests.RequestException:
            pass

        # Fallback to /api/generate if unsuccessful.
        try:
            gen_payload = {"model": model, "stream": False, "prompt": prompt, "options": {"num_predict": max_tokens}}
            r2 = requests.post(GENERATE_ENDPOINT, data=json.dumps(gen_payload), headers={"Content-Type": "application/json"})
            if r2.status_code == 200:
                return r2.json().get("response", "").strip()
            if (r2.status_code == 404) and (not tried_autopull):
                tried_autopull = True
                _autopull_model(model)
                continue
            r2.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"processor: Ollama request failed: {e}")
