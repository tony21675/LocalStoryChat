#!/usr/bin/env python3
import json
import os
import re
import threading
import time
import socket
import subprocess
import shutil
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import pexpect
except Exception as exc:
    raise SystemExit(
        "Local Story Chat requires the Python 'pexpect' package. Error: " + str(exc)
    )

HOST = "127.0.0.1"
PORT = 8787
BASE = Path.home()

LLAMA = Path(os.path.expanduser(
    "~/Documents/llama.cpp/build/bin/llama-cli"
))

MODELS_DIR = BASE / "Documents" / "Models"

DEFAULT_MODEL = Path(os.path.expanduser(
    "~/Documents/Models/llmfan46--gemma-4-E4B-it-ultra-uncensored-heretic-GGUF--gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
))

STORY_DIR = BASE / "Documents" / "StoryJSON"

STORY_NAMES = [
    "Tony.json",
    "Tiffany.json",
    "Maya.json",
    "Chloe.json",
    "story_bible.json",
    "current_state.json",
]

SYSTEM_PROMPT = r'''You are the story generation engine for an ongoing fictional story.

The six attached JSON files are PRIVATE REFERENCE MATERIAL. Use them silently to maintain continuity. Never quote, dump, summarize, expose, or discuss the contents of the files unless the user explicitly asks for that information.

CANON AUTHORITY:
1. Tony.json, Tiffany.json, Maya.json, and Chloe.json control those characters' established identities, appearance, personality, relationships, background, skills, and knowledge.
2. story_bible.json controls permanent story canon, required events, hidden canon, and continuity rules.
3. current_state.json controls the exact current story situation, locations, recent events, character knowledge, clues, objectives, and immediate continuity.
4. Established story events take priority over guesses or assumptions.

KNOWLEDGE:
- Characters know only what they personally witnessed, experienced, were told, or could reasonably infer.
- Never give a character information marked unknown.
- Keep hidden canon hidden until the story naturally reveals it.
- Do not reveal secrets simply because they appear in the reference files.
- Do not make characters aware of another character's private thoughts, feelings, or secrets unless they have learned them.

CONTINUITY:
- Start from the exact current_state.json situation.
- Preserve established names, ages, appearances, relationships, locations, timeline, objects, knowledge, and required events.
- Do not contradict established canon.
- Do not prematurely resolve unknowns.
- Do not invent major plot facts, locations, people, evidence, motives, or backstory unless the current scene establishes them.
- When something is intentionally unknown, keep it unknown.
- Preserve required emotional reactions and scene order.
- Do not have a character witness something that character did not witness.

WRITING MODE:
- When asked to write or continue the story, output only the story prose unless the user explicitly requests another format.
- Do not output JSON, reference-file contents, analysis, notes, summaries, or explanations during normal story writing.
- Do not repeat previous paragraphs.
- Continue from the exact point requested.
- Do not restart a scene unless explicitly asked.
- Use natural dialogue and consistent character behavior.
- Keep the writing focused on the current scene.

Before generating, silently check the requested scene against the reference files and current state for continuity.'''

STATE_PATH = STORY_DIR / "current_state.json"
STATE_BACKUP_DIR = STORY_DIR / "state_backups"

STATE_REQUIRED_KEYS = {
    "status",
    "chapter",
    "scene",
    "scene_completed",
    "location",
    "time",
    "current_situation",
    "character_knowledge",
    "completed_events",
    "active_clues",
    "new_clues",
    "unresolved_questions",
    "active_objectives",
    "continuity_requirements",
}

STATE_SYSTEM_PROMPT = r"""You are the continuity manager for an ongoing fictional story.

Your job is to identify ONLY the changes caused by a completed story section.

Return ONLY one valid JSON object.

IMPORTANT:
- Return a PARTIAL UPDATE, not a complete current_state.json.
- Include only top-level fields that changed.
- Do not repeat unchanged fields.
- For changed nested objects, include only changed nested keys.
- For arrays, return the complete replacement array only when that array changed.
- Record only events that actually happened in the supplied story section.
- Never invent future events.
- Never turn an unknown fact into a known fact.
- Keep character knowledge limited to what the character could actually know.
- Do not reveal hidden canon.
- Do not invent clues, locations, motives, identities, evidence, relationships, or backstory.
- If nothing changed in a field, omit it.
- The application will merge your partial update into the existing current_state.json.
- Do not output markdown, explanations, notes, analysis, or code fences.
"""

pending_state = None


def read_current_state():
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"current_state.json not found: {STATE_PATH}"
        )

    return json.loads(
        STATE_PATH.read_text(encoding="utf-8")
    )


def validate_state(data):
    if not isinstance(data, dict):
        raise ValueError(
            "State must be a JSON object."
        )

    missing = sorted(
        STATE_REQUIRED_KEYS - set(data.keys())
    )

    if missing:
        raise ValueError(
            "State is missing required keys: "
            + ", ".join(missing)
        )

    if not isinstance(data.get("chapter"), int):
        raise ValueError(
            "State field 'chapter' must be an integer."
        )

    if not isinstance(data.get("scene"), int):
        raise ValueError(
            "State field 'scene' must be an integer."
        )

    if not isinstance(data.get("scene_completed"), bool):
        raise ValueError(
            "State field 'scene_completed' must be true or false."
        )

    if not isinstance(
        data.get("character_knowledge"),
        dict
    ):
        raise ValueError(
            "State field 'character_knowledge' must be an object."
        )

    if not isinstance(
        data.get("current_situation"),
        str
    ):
        raise ValueError(
            "State field 'current_situation' must be a string."
        )

    return data


def merge_state(base, patch):
    if not isinstance(base, dict):
        raise ValueError(
            "Base state must be a JSON object."
        )

    if not isinstance(patch, dict):
        raise ValueError(
            "State update must be a JSON object."
        )

    result = dict(base)

    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = merge_state(
                result[key],
                value
            )
        else:
            result[key] = value

    return result


def extract_json_object(text):
    decoder = json.JSONDecoder()
    text = text or ""

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                text[index:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            return obj

    raise ValueError(
        "The state manager did not return a JSON object."
    )


def generate_state_proposal(story_text):
    global pending_state

    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    current_state = read_current_state()

    prompt = f"""Identify ONLY the changes caused by this completed story section.

CURRENT STATE BEFORE THIS SECTION:
{json.dumps(current_state, indent=2, ensure_ascii=False)}

COMPLETED STORY SECTION:
{story_text}

Return ONLY a JSON object containing changed fields.
Return {{}} if nothing changed.
Do not return the complete current_state.json.
"""

    env = os.environ.copy()

    env["LD_LIBRARY_PATH"] = (
        str(LLAMA.parent)
        + (
            ":" + env["LD_LIBRARY_PATH"]
            if env.get("LD_LIBRARY_PATH")
            else ""
        )
    )

    args = [
        str(LLAMA),
        "-m", str(session.model_path),
        "-ngl", "0",
        "--device", "none",
        "-c", "12288",
        "--reasoning", "off",
        "--temp", "0.10",
        "--top-k", "20",
        "--top-p", "0.80",
        "--repeat-last-n", "256",
        "--repeat-penalty", "1.08",
        "--n-predict", "1000",
        "--system-prompt", STATE_SYSTEM_PROMPT,
        "--prompt", prompt,
        "--color", "off",
        "--no-display-prompt",
        "--simple-io",
        "--single-turn",
    ]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        output, _ = proc.communicate(
            timeout=600
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()

        raise TimeoutError(
            "State update generation timed out."
        )

    if proc.returncode not in (0, None):
        raise RuntimeError(
            f"State manager exited with code {proc.returncode}.\\n"
            f"{output[-2000:]}"
        )


    patch = extract_json_object(output)

    if not isinstance(patch, dict):
        raise ValueError(
            "The state manager did not return a JSON object."
        )

    # Prevent the model from accidentally creating character/location
    # fields at the top level of current_state.json.
    for misplaced in ("Tony", "Tiffany", "Maya", "Chloe"):
        patch.pop(misplaced, None)

    proposed = merge_state(
        current_state,
        patch
    )

    # new_clues describes only clues introduced by THIS section.
    # Never carry old section clues forward as "new".
    proposed["new_clues"] = patch.get("new_clues", [])

    validate_state(proposed)

    pending_state = proposed

    return proposed


def resolve_model_path(value):
    path = Path(os.path.expanduser(str(value))).resolve()
    models_root = MODELS_DIR.resolve()

    try:
        path.relative_to(models_root)
    except ValueError as exc:
        raise ValueError(
            "Model must be inside ~/Documents/Models"
        ) from exc

    if path.suffix.lower() != ".gguf":
        raise ValueError(
            "Selected model is not a GGUF file"
        )

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return path


def available_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    result = []

    for path in sorted(
        MODELS_DIR.rglob("*.gguf"),
        key=lambda p: p.name.lower()
    ):
        result.append({
            "name": path.name,
            "path": str(path)
        })

    return result


def load_story_files(names=None):
    names = STORY_NAMES if names is None else names

    unknown = [name for name in names if name not in STORY_NAMES]
    if unknown:
        raise ValueError(
            "Unknown story file(s): " + ", ".join(unknown)
        )

    result = []

    for name in names:
        path = STORY_DIR / name

        if path.exists():
            try:
                result.append(
                    (name, path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                result.append(
                    (
                        name,
                        json.dumps(
                            {"_error": str(exc)},
                            ensure_ascii=False
                        )
                    )
                )

    return result


def load_default_story_files():
    return load_story_files(STORY_NAMES)


class LlamaSession:
    def __init__(self):
        self.child = None
        self.lock = threading.RLock()
        self.system_prompt = SYSTEM_PROMPT
        self.files = []
        self.started_at = None
        self.buffer = ""
        self.model_path = DEFAULT_MODEL.resolve() if DEFAULT_MODEL.exists() else None

    def _stop_unlocked(self):
        proc = self.child
        self.child = None

        if proc is None:
            return

        try:
            if proc.stdin:
                proc.stdin.write(b"/exit\n")
                proc.stdin.flush()
        except Exception:
            pass

        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        self.buffer = ""

    def stop(self):
        with self.lock:
            self._stop_unlocked()

    def _build_prompt(self, base_prompt, files):
        parts = [base_prompt.strip()]

        if files:
            parts.append("\nAUTHORITATIVE STORY FILES START HERE\n")

            for name, content in files:
                parts.append(f"\n[Attached File: {name}]\n{content}\n")

            parts.append("\nAUTHORITATIVE STORY FILES END HERE\n")

        return "\n".join(parts)

    def _read_more(self, timeout):
        if self.child is None or self.child.stdout is None:
            raise RuntimeError("llama-cli stdout is unavailable")

        import select

        ready, _, _ = select.select([self.child.stdout], [], [], timeout)

        if not ready:
            return ""

        chunk = os.read(self.child.stdout.fileno(), 8192)

        if not chunk:
            return ""

        return chunk.decode("utf-8", errors="replace")

    def _drain_pending_output(self, quiet_window=0.25):
        if self.child is None or self.child.stdout is None:
            return

        import select

        deadline = time.time() + quiet_window

        while time.time() < deadline:
            timeout = min(
                0.05,
                max(0.01, deadline - time.time())
            )

            ready, _, _ = select.select(
                [self.child.stdout],
                [],
                [],
                timeout
            )

            if not ready:
                break

            chunk = os.read(
                self.child.stdout.fileno(),
                8192
            )

            if not chunk:
                break

    def _wait_for_prompt(self, timeout, initial=False):
        deadline = time.time() + timeout
        text = self.buffer
        prompt_index = -1

        while time.time() < deadline:
            if initial:
                prompt_index = text.find("> ")
            else:
                prompt_index = text.rfind("\n> ")

                if prompt_index < 0 and text.startswith("> "):
                    prompt_index = 0

            if prompt_index >= 0:
                before = text[:prompt_index]

                after = text[
                    prompt_index + (2 if prompt_index == 0 else 3):
                ]

                self.buffer = after
                return before

            chunk = self._read_more(
                min(
                    1.0,
                    max(0.05, deadline - time.time())
                )
            )

            if chunk:
                text += chunk
                continue

            if self.child and self.child.poll() is not None:
                tail = text[-1600:]

                raise RuntimeError(
                    "llama-cli exited unexpectedly "
                    f"(code {self.child.returncode}). "
                    f"Backend output:\n{tail}"
                )

        raise TimeoutError(
            "Timed out waiting for llama-cli prompt. "
            f"Recent backend output:\n{text[-1600:]}"
        )

    @staticmethod
    def clean_output(text):
        lines = []

        for line in (text or "").replace("\r", "").splitlines():
            s = line.strip()

            if not s:
                if lines and lines[-1] != "":
                    lines.append("")
                continue

            if s.startswith("[ Prompt:") or s.startswith("[ Generation:"):
                continue

            lines.append(line)

        return "\n".join(lines).strip()

    def start(self, system_prompt, files, model_path=None):
        with self.lock:
            self._stop_unlocked()

            if not LLAMA.exists():
                raise FileNotFoundError(
                    f"llama-cli not found: {LLAMA}"
                )

            if model_path is not None:
                model_path = resolve_model_path(model_path)
                self.model_path = model_path

            if self.model_path is None:
                raise RuntimeError(
                    "No model is loaded. Use Load Model first."
                )

            self.system_prompt = system_prompt
            self.files = files

            full_prompt = self._build_prompt(
                system_prompt,
                files
            )

            env = os.environ.copy()

            env["LD_LIBRARY_PATH"] = (
                str(LLAMA.parent)
                + (
                    ":" + env["LD_LIBRARY_PATH"]
                    if env.get("LD_LIBRARY_PATH")
                    else ""
                )
            )

            args = [
                str(LLAMA),
                "-m", str(self.model_path),
                "-ngl", "0",
                "--device", "none",
                "-c", "12288",
                "--reasoning", "off",
                "--repeat-last-n", "256",
                "--repeat-penalty", "1.08",
                "--n-predict", "2000",
                "--system-prompt", full_prompt,
                "--color", "off",
                "--no-display-prompt",
                "--simple-io",
            ]

            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=0,
            )

            self.child = proc
            self.started_at = time.time()
            self.buffer = ""

            try:
                self._wait_for_prompt(
                    180,
                    initial=True
                )

                self.buffer = ""
                self._drain_pending_output()

            except Exception:
                tail = self.buffer[-1600:]
                self._stop_unlocked()

                raise RuntimeError(
                    "llama-cli did not become ready. "
                    f"Backend output:\n{tail}"
                )

    def ask(self, text):
        with self.lock:
            proc = self.child

            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    "Session is not running. Click New Chat to start it."
                )

            started = time.time()

            try:
                proc.stdin.write(
                    (text + "\n").encode("utf-8")
                )

                proc.stdin.flush()

                raw = self._wait_for_prompt(
                    900,
                    initial=False
                )

                answer = self.clean_output(raw)

                if not answer:
                    raise RuntimeError(
                        "llama-cli returned an empty response"
                    )

                return answer, time.time() - started

            except BrokenPipeError as exc:
                raise RuntimeError(
                    "The llama-cli session closed its input pipe."
                ) from exc

            except OSError as exc:
                raise RuntimeError(str(exc)) from exc

    def estimate_context(self, extra=""):
        total = (
            len(self.system_prompt)
            + sum(len(c) for _, c in self.files)
            + len(extra)
        )

        return max(
            0,
            (total + 3) // 4
        )


session = LlamaSession()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, data):
        raw = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        try:
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.send_header(
                "Content-Length",
                str(len(raw))
            )
            self.send_header(
                "Cache-Control",
                "no-store"
            )
            self.end_headers()
            self.wfile.write(raw)

        except BrokenPipeError:
            pass

    def _body(self):
        n = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        return json.loads(
            self.rfile.read(n) or b"{}"
        )

    def _restart_with_current_state(self):
        if session.model_path is None:
            raise RuntimeError(
                "No model is loaded. Use Load Model first."
            )

        session.start(
            session.system_prompt or SYSTEM_PROMPT,
            session.files,
            session.model_path
        )

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/status":
            self._json(200, {
                "running": bool(
                    session.child
                    and session.child.poll() is None
                ),
                "model": (
                    str(session.model_path)
                    if session.model_path
                    else None
                ),
                "llama": str(LLAMA),
                "files": [
                    {
                        "name": name,
                        "loaded": True
                    }
                    for name, _ in session.files
                ],
                "available_json": [
                    {
                        "name": name,
                        "exists": (STORY_DIR / name).exists()
                    }
                    for name in STORY_NAMES
                ],
                "available_models": available_models(),
                "story_dir": str(STORY_DIR),
                "models_dir": str(MODELS_DIR),
                "context_size": 12288,
                "gpu_layers": 0,
                "reasoning": "off",
                "n_predict": 2000,
                "repeat_penalty": 1.08,
                "state": {
                    "chapter": (
                        read_current_state().get("chapter")
                        if STATE_PATH.exists() else None
                    ),
                    "scene": (
                        read_current_state().get("scene")
                        if STATE_PATH.exists() else None
                    ),
                    "status": (
                        read_current_state().get("status")
                        if STATE_PATH.exists() else None
                    ),
                    "pending": pending_state is not None,
                },
            })

            return

        if path == "/api/models":
            self._json(
                200,
                {
                    "models": available_models(),
                    "selected": (
                        str(session.model_path)
                        if session.model_path
                        else None
                    )
                }
            )

            return

        if path == "/api/default-files":
            files = load_default_story_files()

            self._json(
                200,
                {
                    "files": [
                        {
                            "name": name,
                            "content": content
                        }
                        for name, content in files
                    ]
                }
            )

            return

        if path == "/api/json-files":
            self._json(
                200,
                {
                    "files": [
                        {
                            "name": name,
                            "exists": (STORY_DIR / name).exists(),
                            "loaded": any(
                                loaded_name == name
                                for loaded_name, _ in session.files
                            )
                        }
                        for name in STORY_NAMES
                    ]
                }
            )

            return

        if path == "/":
            path = "/index.html"

        file_path = (
            Path(__file__).parent
            / "static"
            / path.lstrip("/")
        )

        if file_path.exists() and file_path.is_file():
            if file_path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif file_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            else:
                content_type = "application/octet-stream"

            raw = file_path.read_bytes()

            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    content_type
                )
                self.send_header(
                    "Content-Length",
                    str(len(raw))
                )
                self.send_header(
                    "Cache-Control",
                    "no-store, no-cache, must-revalidate"
                )
                self.end_headers()
                self.wfile.write(raw)

            except BrokenPipeError:
                pass

            return

        self._json(
            404,
            {
                "error": "Not found"
            }
        )

    def do_POST(self):
        global pending_state
        path = urlparse(self.path).path

        try:
            body = self._body()

            if path == "/api/model/load":
                requested = str(
                    body.get("path", "")
                ).strip()

                if not requested:
                    raise ValueError(
                        "No model was selected."
                    )

                model_path = resolve_model_path(
                    requested
                )

                session.model_path = model_path

                if session.child is not None:
                    self._restart_with_current_state()

                self._json(
                    200,
                    {
                        "ok": True,
                        "model": str(model_path),
                        "running": bool(
                            session.child
                            and session.child.poll() is None
                        )
                    }
                )

                return

            if path == "/api/model/unload":
                session.stop()
                session.model_path = None

                self._json(
                    200,
                    {
                        "ok": True
                    }
                )

                return

            if path == "/api/json/load":
                names = body.get("names")

                if names is None:
                    names = STORY_NAMES

                names = [
                    str(name)
                    for name in names
                ]

                session.files = load_story_files(
                    names
                )

                if session.child is not None:
                    self._restart_with_current_state()

                self._json(
                    200,
                    {
                        "ok": True,
                        "files": [
                            name
                            for name, _ in session.files
                        ],
                        "running": bool(
                            session.child
                            and session.child.poll() is None
                        )
                    }
                )

                return

            if path == "/api/json/unload":
                session.stop()
                session.files = []

                self._json(
                    200,
                    {
                        "ok": True
                    }
                )

                return

            if path == "/api/new-chat":
                if session.model_path is None:
                    raise RuntimeError(
                        "No model is loaded. Use Load Model first."
                    )

                selected_names = [
                    name
                    for name, _ in session.files
                ]

                if selected_names:
                    session.files = load_story_files(selected_names)
                else:
                    session.files = load_default_story_files()

                session.start(
                    session.system_prompt or SYSTEM_PROMPT,
                    session.files,
                    session.model_path
                )

                self._json(
                    200,
                    {
                        "ok": True,
                        "model": str(session.model_path),
                        "files": [
                            name
                            for name, _ in session.files
                        ],
                        "estimated_system_tokens": (
                            session.estimate_context()
                        ),
                        "context_size": 12288
                    }
                )

                return

            if path == "/api/start":
                files = []

                for item in body.get("files", []):
                    name = str(
                        item.get(
                            "name",
                            "unnamed.json"
                        )
                    )[:120]

                    content = str(
                        item.get(
                            "content",
                            ""
                        )
                    )

                    files.append(
                        (name, content)
                    )

                if not files:
                    files = (
                        session.files
                        if session.files
                        else load_default_story_files()
                    )

                system_prompt = str(
                    body.get(
                        "systemPrompt",
                        SYSTEM_PROMPT
                    )
                )

                requested_model = body.get(
                    "model"
                )

                if requested_model:
                    session.model_path = (
                        resolve_model_path(
                            requested_model
                        )
                    )

                session.start(
                    system_prompt,
                    files,
                    session.model_path
                )

                self._json(
                    200,
                    {
                        "ok": True,
                        "model": str(session.model_path),
                        "files": [
                            name
                            for name, _ in files
                        ],
                        "estimated_system_tokens": (
                            session.estimate_context()
                        ),
                        "context_size": 12288
                    }
                )

                return

            if path == "/api/state":
                current = read_current_state()
                self._json(200, {
                    "ok": True,
                    "current": current,
                    "pending": pending_state,
                })
                return

            if path == "/api/state/propose":
                story = str(body.get("story", "")).strip()

                if len(story) < 40:
                    raise ValueError(
                        "The story section is too short to create a useful state update."
                    )

                proposed = generate_state_proposal(story)

                self._json(200, {
                    "ok": True,
                    "state": proposed,
                })
                return

            if path == "/api/state/apply":
                if pending_state is None:
                    raise RuntimeError("There is no pending state update to apply.")

                validate_state(pending_state)

                STATE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

                if STATE_PATH.exists():
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    backup = STATE_BACKUP_DIR / f"current_state-{stamp}.json"
                    shutil.copy2(STATE_PATH, backup)

                STATE_PATH.write_text(
                    json.dumps(
                        pending_state,
                        indent=2,
                        ensure_ascii=False
                    ) + "\n",
                    encoding="utf-8"
                )

                pending_state = None

                self._json(200, {
                    "ok": True,
                    "message": (
                        "current_state.json applied. "
                        "Start New Chat to load the updated state into the model."
                    ),
                })
                return

            if path == "/api/state/discard":
                pending_state = None

                self._json(200, {
                    "ok": True
                })
                return

            if path == "/api/chat":
                text = str(
                    body.get(
                        "message",
                        ""
                    )
                ).strip()

                if not text:
                    self._json(
                        400,
                        {
                            "error": "Message is empty"
                        }
                    )

                    return

                started = time.time()

                answer, measured = session.ask(
                    text
                )

                elapsed = measured

                self._json(
                    200,
                    {
                        "ok": True,
                        "content": answer,
                        "seconds": round(
                            elapsed,
                            1
                        ),
                        "estimated_context_tokens": (
                            session.estimate_context(
                                text + "\n" + answer
                            )
                        ),
                        "context_size": 12288
                    }
                )

                return

            if path == "/api/stop":
                session.stop()

                self._json(
                    200,
                    {
                        "ok": True
                    }
                )

                return

            self._json(
                404,
                {
                    "error": "Not found"
                }
            )

        except Exception as exc:
            self._json(
                500,
                {
                    "error": str(exc)
                }
            )

    def log_message(self, fmt, *args):
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{fmt % args}"
        )


def _open_browser_when_ready():
    url = f"http://{HOST}:{PORT}"

    for _ in range(50):
        try:
            with socket.create_connection(
                (HOST, PORT),
                timeout=0.25
            ):
                webbrowser.open(
                    url,
                    new=2
                )

                return

        except OSError:
            time.sleep(0.1)


def main():
    httpd = ThreadingHTTPServer(
        (HOST, PORT),
        Handler
    )

    print(
        f"Local Story Chat: http://{HOST}:{PORT}"
    )

    print(
        f"llama-cli: {LLAMA}"
    )

    print(
        f"default model: {DEFAULT_MODEL}"
    )

    print(
        f"models directory: {MODELS_DIR}"
    )

    print(
        "Browser will open automatically."
    )

    print(
        "Press Ctrl+C to stop the web app."
    )

    threading.Thread(
        target=_open_browser_when_ready,
        name="browser-opener",
        daemon=True
    ).start()

    try:
        httpd.serve_forever()

    finally:
        session.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
