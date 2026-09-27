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
APP_ROOT = Path(__file__).resolve().parent

LLAMA = Path(os.path.expanduser(
    "~/Documents/llama.cpp/build/bin/llama-cli"
))

MODELS_DIR = BASE / "Documents" / "Models"

DEFAULT_MODEL = Path(os.path.expanduser(
    "~/Documents/Models/llmfan46--gemma-4-E4B-it-ultra-uncensored-heretic-GGUF--gemma-4-E4B-it-ultra-uncensored-heretic-Q4_K_M.gguf"
))

STORIES_ROOT = APP_ROOT / "stories"
ACTIVE_STORY_FILE = STORIES_ROOT / ".active_story"


def available_story_dirs():
    STORIES_ROOT.mkdir(parents=True, exist_ok=True)
    return sorted(
        [
            path
            for path in STORIES_ROOT.iterdir()
            if path.is_dir()
            and (path / "story_bible.json").is_file()
        ],
        key=lambda p: p.name.lower()
    )


def _validate_story_dir(path):
    path = Path(path).expanduser().resolve()
    root = STORIES_ROOT.resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Story folder must be inside the application stories directory"
        ) from exc

    if not path.is_dir():
        raise ValueError(f"Story folder does not exist: {path}")

    if not (path / "story_bible.json").is_file():
        raise ValueError(
            f"Story folder is missing story_bible.json: {path}"
        )

    if not (path / "current_state.json").is_file():
        raise ValueError(
            f"Story folder is missing current_state.json: {path}"
        )

    return path


def set_active_story_dir(path, persist=True):
    global STORY_DIR, MANUSCRIPT_DIR, STATE_PATH, STATE_BACKUP_DIR

    path = _validate_story_dir(path)

    STORY_DIR = path
    MANUSCRIPT_DIR = path / "Manuscript"
    STATE_PATH = path / "current_state.json"
    STATE_BACKUP_DIR = path / "state_backups"

    if persist:
        STORIES_ROOT.mkdir(parents=True, exist_ok=True)
        ACTIVE_STORY_FILE.write_text(
            path.name + "\n",
            encoding="utf-8"
        )

    return STORY_DIR


def discover_initial_story_dir():
    stories = available_story_dirs()

    if ACTIVE_STORY_FILE.is_file():
        name = ACTIVE_STORY_FILE.read_text(
            encoding="utf-8"
        ).strip()

        if name:
            candidate = (STORIES_ROOT / name).resolve()

            try:
                return _validate_story_dir(candidate)
            except ValueError:
                pass

    if len(stories) == 1:
        return stories[0]

    if not stories:
        raise FileNotFoundError(
            "No story folders found in the application stories directory"
        )

    raise RuntimeError(
        "Multiple story folders exist, but no active story is selected."
    )


STORY_DIR = discover_initial_story_dir()
MANUSCRIPT_DIR = STORY_DIR / "Manuscript"
STATE_PATH = STORY_DIR / "current_state.json"
STATE_BACKUP_DIR = STORY_DIR / "state_backups"

def discover_story_names():
    """Discover the active story files from story_bible.json.

    The application itself does not know any character names or plot details.
    story_bible.json declares the character-card files for the current novel.
    current_state.json is always included as the live story state.
    """
    story_bible = STORY_DIR / "story_bible.json"
    current_state = STORY_DIR / "current_state.json"

    if not story_bible.exists():
        raise FileNotFoundError(
            f"story_bible.json not found: {story_bible}"
        )

    try:
        data = json.loads(
            story_bible.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(
            f"Could not read story_bible.json: {exc}"
        ) from exc

    names = ["story_bible.json"]

    character_cards = data.get("character_cards", [])

    if not isinstance(character_cards, list):
        raise ValueError(
            "story_bible.json character_cards must be a list"
        )

    for item in character_cards:
        name = str(item).strip()
        if not name:
            continue
        if name not in names:
            names.append(name)

    if current_state.exists():
        names.append("current_state.json")

    return names

SYSTEM_PROMPT = r'''You are writing an ongoing fictional novel.

Use the supplied story context as canon and continuity reference.

Character files define established character identity, personality, relationships, background, and important character facts.
story_bible.json defines established world and story canon.
current_state.json defines the present situation, locations, knowledge, important recent events, and what has not happened yet.
The user's current request defines what to write now.
The previous saved manuscript section is the direct prose bridge from the preceding scene.

Keep established facts consistent. Characters only know what they could reasonably know from the story.

Ordinary conversation is normal. Characters can joke, gossip, talk about school, plans, food, family, hobbies, memories, and other everyday things. Most casual conversation is scene-only and does not need to become permanent story memory. Important events, discoveries, lasting decisions, consequential knowledge, meaningful relationship changes, important plans, clues, and other facts that later scenes need to remember are the things worth preserving.

Use the requested characters and let them behave like real people. Follow the scene goal, required events, boundaries, tone, and requested length.

When the scene includes a required event, actually show it happening in dialogue, action, and reaction. Do not merely describe that it happened. If the request describes a revealing slip, accidental confession, or similar dialogue sequence without naming the exact wording, use the most directly established character fact that the scene goal and canon clearly point toward. Do not substitute a different revelation.

Start from the actual current story position. Do not invent an earlier conversation to respond to. Do not restart the previous saved section.

Protected scene boundaries are literal. When the request says there is no threat, suspicious encounter, or major plot event yet, do not manufacture suspense by having characters stop, listen for something, scan their surroundings, become watchful, or react to an unexplained sound. Keep ordinary moments ordinary until an actual established event changes the situation.

Write a complete scene with a natural beginning, development, and ending. Do not stop just because the characters have exchanged a few lines.

Scene cast and physical presence are a hard boundary. Only characters explicitly listed by the current scene request are physically present and participating in the scene. Other established characters may be mentioned or discussed naturally when appropriate, but their presence elsewhere in current_state.json does not place them in the scene. Never move an excluded character into the scene unless the user explicitly adds that character to the scene request.

Output only the story prose.'''


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

STATE_SYSTEM_PROMPT = r"""You are a minimal continuity updater for an ongoing fictional story.

Your only job is to record persistent story changes that will matter in later sections.

Return ONLY this JSON object:
{
  "patch": {},
  "evidence": []
}

DEFAULT TO NO UPDATE.
For ordinary dialogue, jokes, school talk, small observations, temporary emotions, gestures, routine movement, and disposable scene details, return:
{"patch": {}, "evidence": []}

Never copy or summarize current_state.json.
Never repeat unchanged information.
Never repeat existing array items.
Never add continuity_requirements unless a genuinely new persistent fact or rule was established.
Never add active_objectives unless a meaningful lasting objective changed.
Never add character_knowledge unless a character clearly learned a new consequential fact.
Never add completed_events for ordinary conversation topics or incidental actions.
Never add active_clues or unresolved_questions unless a real plot clue or unresolved plot question was established.

Allowed top-level patch fields:
status, chapter, scene, scene_completed, location, time, current_situation,
character_knowledge, completed_events, active_clues, new_clues,
unresolved_questions, active_objectives, continuity_requirements.

For arrays, include ONLY new items introduced by this section.

Every substantive patch item needs one short exact contiguous evidence quote copied from the completed story section. If it cannot be directly quoted, omit the change.

Chapter and scene numbers are application bookkeeping. Do not infer scene number from the manuscript section number.

Keep the response extremely small. An empty patch is often the correct answer.
Do not output markdown, explanations, notes, analysis, or code fences.
"""

pending_state = None
STORY_SAVE_LOCK = threading.Lock()


def next_story_section_path(chapter):
    chapter_dir = MANUSCRIPT_DIR / "Chapters" / f"Chapter_{chapter:02d}"
    highest_section = 0

    if chapter_dir.is_dir():
        for path in chapter_dir.glob(
            f"Chapter_{chapter:02d}_Section_*.txt"
        ):
            match = re.search(
                r"_Section_(\d+)\.txt$",
                path.name
            )
            if not match:
                continue

            try:
                number = int(match.group(1))
            except ValueError:
                continue

            highest_section = max(highest_section, number)

    candidate_section = highest_section + 1
    filename = (
        f"Chapter_{chapter:02d}_Section_{candidate_section:02d}.txt"
    )
    path_out = chapter_dir / filename

    while path_out.exists():
        candidate_section += 1
        filename = (
            f"Chapter_{chapter:02d}_Section_{candidate_section:02d}.txt"
        )
        path_out = chapter_dir / filename

    return candidate_section, filename, path_out


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
        existing = result.get(key)

        if (
            isinstance(value, dict)
            and isinstance(existing, dict)
        ):
            result[key] = merge_state(
                existing,
                value
            )

        elif (
            isinstance(value, list)
            and isinstance(existing, list)
        ):
            merged = list(existing)

            for item in value:
                if item not in merged:
                    merged.append(item)

            result[key] = merged

        else:
            result[key] = value

    return result


_STATE_METADATA_FIELDS = {
    "chapter",
    "scene",
    "scene_completed",
    "status",
}


def _normalize_evidence_text(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "")
    ).strip()


def _claim_key(value):
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False
    )


def _iter_changed_claims(base, patch, path=""):
    if isinstance(patch, dict):
        base_dict = base if isinstance(base, dict) else {}

        for key, value in patch.items():
            child_path = (
                f"{path}.{key}"
                if path
                else key
            )

            yield from _iter_changed_claims(
                base_dict.get(key),
                value,
                child_path
            )

        return

    if isinstance(patch, list):
        base_list = base if isinstance(base, list) else []

        for item in patch:
            if item not in base_list:
                yield path, item

        return

    if patch != base:
        yield path, patch


def _filter_patch_to_supported_evidence(base, patch, story_text, evidence):
    """Keep only substantive patch claims supported by exact story quotes.

    The state manager is a local language model and may occasionally emit an
    imprecise or stitched evidence quote. A single bad quote should not discard
    an otherwise useful state update. Unsupported claims are removed from the
    proposed patch; authoritative bookkeeping fields remain eligible.
    """
    normalized_story = _normalize_evidence_text(story_text)

    valid_evidence = set()

    if isinstance(evidence, list):
        for entry in evidence:
            if not isinstance(entry, dict):
                continue

            field = entry.get("field")
            claim = entry.get("claim")
            quote = entry.get("quote")

            if not isinstance(field, str) or not field.strip():
                continue

            if "claim" not in entry:
                continue

            if not isinstance(quote, str) or not quote.strip():
                continue

            normalized_quote = _normalize_evidence_text(quote)

            if normalized_quote not in normalized_story:
                continue

            valid_evidence.add(
                (
                    field.strip(),
                    _claim_key(claim)
                )
            )

    def keep_value(base_value, value, path):
        top_level = path.split(".", 1)[0]

        if top_level in _STATE_METADATA_FIELDS:
            return True, value

        if isinstance(value, dict):
            kept = {}

            for key, child in value.items():
                child_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )

                child_base = (
                    base_value.get(key)
                    if isinstance(base_value, dict)
                    else None
                )

                keep, child_value = keep_value(
                    child_base,
                    child,
                    child_path
                )

                if keep:
                    kept[key] = child_value

            return bool(kept), kept

        if isinstance(value, list):
            base_list = (
                base_value
                if isinstance(base_value, list)
                else []
            )

            kept = []

            for item in value:
                item_key = (
                    path,
                    _claim_key(item)
                )

                if item in base_list or item_key in valid_evidence:
                    kept.append(item)

            return bool(kept), kept

        if value == base_value:
            return False, None

        return (
            (path, _claim_key(value)) in valid_evidence,
            value
        )

    filtered = {}

    for key, value in patch.items():
        base_value = base.get(key)

        if key in _STATE_METADATA_FIELDS:
            filtered[key] = value
            continue

        keep, filtered_value = keep_value(
            base_value,
            value,
            key
        )

        if keep:
            filtered[key] = filtered_value

    return filtered


def validate_state_patch(base, patch, story_text, evidence):
    if not isinstance(base, dict):
        raise ValueError(
            "State patch validation requires an object as the base state."
        )

    if not isinstance(patch, dict):
        raise ValueError(
            "State patch must be a JSON object."
        )

    if not isinstance(evidence, list):
        raise ValueError(
            "State proposal rejected. 'evidence' must be an array."
        )

    unknown_top_level = sorted(
        set(patch.keys()) - STATE_REQUIRED_KEYS
    )

    if unknown_top_level:
        raise ValueError(
            "State proposal rejected. Unknown top-level fields: "
            + ", ".join(unknown_top_level)
        )

    for key, value in patch.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            new_nested_keys = sorted(
                set(value.keys()) - set(base[key].keys())
            )

            if new_nested_keys:
                raise ValueError(
                    f"State proposal rejected. Field '{key}' "
                    "contains new nested keys: "
                    + ", ".join(new_nested_keys)
                )

    normalized_story = _normalize_evidence_text(
        story_text
    )

    changed_claims = {
        (field, _claim_key(claim))
        for field, claim in _iter_changed_claims(base, patch)
        if field.split(".", 1)[0] not in _STATE_METADATA_FIELDS
    }

    evidence_keys = set()

    for entry in evidence:
        # The model may occasionally emit malformed or incomplete evidence
        # records for unchanged facts. Ignore those records; substantive
        # changed claims still require at least one valid evidence match.
        if not isinstance(entry, dict):
            continue

        field = entry.get("field")
        claim = entry.get("claim")
        quote = entry.get("quote")

        if not isinstance(field, str) or not field.strip():
            continue

        if "claim" not in entry:
            continue

        if not isinstance(quote, str) or not quote.strip():
            continue

        key = (
            field.strip(),
            _claim_key(claim)
        )

        # The state model may echo evidence for unchanged facts from the
        # baseline. Those are not state changes and should not block an
        # otherwise valid proposal. Only evidence attached to a genuinely
        # changed claim is validated.
        if key not in changed_claims:
            continue

        normalized_quote = _normalize_evidence_text(
            quote
        )

        if normalized_quote not in normalized_story:
            raise ValueError(
                "State proposal rejected. Evidence quote was not found "
                "in the completed story section: "
                f"{quote!r}"
            )

        evidence_keys.add(key)

    for key in changed_claims:
        field = key[0]
        claim = next(
            claim
            for changed_field, changed_claim in _iter_changed_claims(base, patch)
            if changed_field == field
            and _claim_key(changed_claim) == key[1]
        )

        if key not in evidence_keys:
            raise ValueError(
                "State proposal rejected. Missing exact evidence for "
                f"changed field '{field}': {claim!r}"
            )


def extract_json_object(text):
    decoder = json.JSONDecoder()
    text = text or ""

    # llama-cli can echo part of the prompt before the actual response.
    # Search every JSON object and keep the last object with the exact state
    # proposal keys. Do not require the whole stdout stream to be JSON.
    candidates = []

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
            candidates.append(obj)

    for obj in reversed(candidates):
        if (
            "patch" in obj
            and "evidence" in obj
            and isinstance(obj.get("patch"), dict)
            and isinstance(obj.get("evidence"), list)
        ):
            return obj

    # Fallback for model output that contains valid JSON after echoed prompt
    # text but has characters before/after it that prevent normal candidate
    # discovery. Start at the final state-object marker and decode from there.
    marker = text.rfind('{"patch"')
    if marker >= 0:
        try:
            obj, _ = decoder.raw_decode(
                text[marker:]
            )

            if (
                isinstance(obj, dict)
                and "patch" in obj
                and "evidence" in obj
                and isinstance(obj.get("patch"), dict)
                and isinstance(obj.get("evidence"), list)
            ):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The state manager did not return a valid proposal with "
        "'patch' and 'evidence'."
    )


def extract_json_with_keys(text, required_keys):
    decoder = json.JSONDecoder()
    text = text or ""

    candidates = []

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
            candidates.append(obj)

    required_keys = set(required_keys)

    for obj in reversed(candidates):
        if required_keys.issubset(obj.keys()):
            return obj

    raise ValueError(
        "The model did not return the required JSON structure."
    )


def _filter_unestablished_location_changes(base, patch, story_text):
    """Do not let state advance physical location without explicit arrival."""
    if not isinstance(patch, dict):
        return patch

    base_location = base.get("location")
    patch_location = patch.get("location")

    if not isinstance(base_location, dict) or not isinstance(patch_location, dict):
        return patch

    normalized_story = _normalize_evidence_text(story_text).lower()

    explicit_arrival = bool(
        re.search(
            r"(?:arrived(?: at| home| inside)|reached (?:home|the house|their house|the front door)|entered (?:the house|their house|the home)|went inside|stepped inside)",
            normalized_story,
        )
    )

    if explicit_arrival:
        return patch

    filtered_location = dict(patch_location)

    for character, old_value in base_location.items():
        if not isinstance(old_value, str):
            continue

        old_text = old_value.lower()

        if "not reached" not in old_text:
            continue

        if character not in filtered_location:
            continue

        new_value = filtered_location.get(character)

        if new_value != old_value:
            filtered_location.pop(character, None)

    if filtered_location:
        patch["location"] = filtered_location
    else:
        patch.pop("location", None)

    base_situation = str(base.get("current_situation", "")).lower()
    proposed_situation = patch.get("current_situation")

    if (
        "not reached" in base_situation
        and isinstance(proposed_situation, str)
        and re.search(
            r"(?:arrived|reached|inside|entered)",
            proposed_situation.lower(),
        )
    ):
        patch.pop("current_situation", None)

    return patch


def generate_state_proposal(
    story_text,
    section_filename=None,
    requested_chapter=None,
    requested_scene=None,
):
    global pending_state

    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    current_state = read_current_state()

    section_chapter = None
    section_number = None

    if section_filename:
        match = re.search(
            r"^Chapter_(\d+)_Section_(\d+)\.txt$",
            str(section_filename).strip(),
            re.IGNORECASE
        )

        if match:
            section_chapter = int(match.group(1))
            section_number = int(match.group(2))

    prompt = f"""Update story state ONLY when this completed story section establishes a persistent change.

CURRENT STATE BEFORE THIS SECTION:
{json.dumps(current_state, ensure_ascii=False, indent=2)}

COMPLETED STORY SECTION:
{story_text}

REQUESTED NARRATIVE POSITION:
Chapter {requested_chapter if isinstance(requested_chapter, int) else current_state.get("chapter")} / Scene {requested_scene if isinstance(requested_scene, int) else current_state.get("scene")}

Return ONLY:
{{"patch": {{}}, "evidence": []}}

Rules:
- Default to an empty patch for ordinary scene material.
- Return only genuinely new or changed persistent information.
- Do not copy, summarize, or restate current_state.
- Do not add ordinary conversation topics, temporary observations, routine actions, or disposable details to state.
- Do not create new clues or questions unless the section clearly makes them plot-relevant.
- For array fields, include only new items.
- Every substantive patch item needs one short exact contiguous quote from the completed story section.
- If a proposed change cannot be directly quoted, omit it.
- The application supplies Chapter/Scene bookkeeping separately. Do not infer scene number from the manuscript section number.
- Keep the response extremely small. Empty patch is preferred when nothing important changed.
- Output no markdown or explanation.
"""

    # Run the state manager only after the writer model has been stopped.
    # This avoids loading a second copy of the model into an 8 GB machine.
    saved_model_path = session.model_path
    saved_system_prompt = session.system_prompt or SYSTEM_PROMPT
    saved_files = list(session.files)

    session.stop()

    try:
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
            "-m", str(saved_model_path),
            "-ngl", "0",
            "--device", "none",
            "-c", "8192",
            "--reasoning", "off",
            "--temp", "0.10",
            "--top-k", "20",
            "--top-p", "0.80",
            "--repeat-last-n", "256",
            "--repeat-penalty", "1.08",
            "--n-predict", "400",
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
                timeout=300
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

        print("\n--- RAW STATE MANAGER OUTPUT ---", flush=True)
        print(output, flush=True)
        print("--- END RAW STATE MANAGER OUTPUT ---\n", flush=True)

        try:
            proposal = extract_json_object(output)
        except ValueError as exc:
            print(
                "State manager returned no usable JSON proposal; "
                "using an empty patch. "
                f"Reason: {exc}",
                flush=True
            )
            proposal = {
                "patch": {},
                "evidence": [],
            }

        if not isinstance(proposal, dict):
            proposal = {
                "patch": {},
                "evidence": [],
            }

        patch = proposal.get("patch")
        evidence = proposal.get("evidence")

        if patch is None:
            raise ValueError(
                "State manager response is missing 'patch'."
            )

        if evidence is None:
            raise ValueError(
                "State manager response is missing 'evidence'."
            )

        if not isinstance(patch, dict):
            raise ValueError(
                "State manager 'patch' must be a JSON object."
            )

        if not isinstance(evidence, list):
            raise ValueError(
                "State manager 'evidence' must be an array."
            )

        unknown_top_level = sorted(
            set(patch.keys()) - STATE_REQUIRED_KEYS
        )

        if unknown_top_level:
            print(
                "Ignoring unsupported state fields: "
                + ", ".join(unknown_top_level),
                flush=True
            )

            for name in unknown_top_level:
                patch.pop(name, None)

        current_characters = current_state.get("character_knowledge", {})

        if isinstance(current_characters, dict):
            for misplaced in current_characters:
                patch.pop(misplaced, None)

        if section_chapter is not None:
            if section_chapter < current_state.get("chapter", section_chapter):
                raise ValueError(
                    "Selected manuscript section is older than the current story state."
                )

        if isinstance(requested_chapter, int):
            if requested_chapter < current_state.get("chapter", requested_chapter):
                raise ValueError(
                    "Requested narrative chapter is earlier than the current story state."
                )
            patch["chapter"] = requested_chapter

        if isinstance(requested_scene, int):
            if requested_scene < 1:
                raise ValueError(
                    "Requested narrative scene must be at least 1."
                )
            patch["scene"] = requested_scene

        patch = _filter_patch_to_supported_evidence(
            current_state,
            patch,
            story_text,
            evidence
        )

        patch = _filter_unestablished_location_changes(
            current_state,
            patch,
            story_text
        )

        validate_state_patch(
            current_state,
            patch,
            story_text,
            evidence
        )

        proposed = merge_state(
            current_state,
            patch
        )

        proposed["new_clues"] = patch.get("new_clues", [])

        validate_state(proposed)

        pending_state = proposed

        return proposed

    finally:
        session.start(
            saved_system_prompt,
            saved_files,
            saved_model_path
        )

def _derive_revelation_guidance(characters):
    """Extract established private relationship facts that may satisfy a reveal."""
    requested = {
        str(item).strip().lower()
        for item in str(characters or "").split(",")
        if str(item).strip()
    }

    if not requested:
        return []

    guidance = []

    for card_name in discover_story_names():
        if not card_name.endswith(".json") or card_name == "current_state.json" or card_name == "story_bible.json":
            continue

        try:
            data = json.loads((STORY_DIR / card_name).read_text(encoding="utf-8"))
        except Exception:
            continue

        name = str(data.get("name", "")).strip()
        if not name or name.lower() not in requested:
            continue

        relationships = data.get("relationships")
        if not isinstance(relationships, dict):
            continue

        for person, value in relationships.items():
            text_value = str(value or "").strip()
            lower_value = text_value.lower()

            if any(term in lower_value for term in (
                "secret crush",
                "crush on",
                "attracted to",
                "has feelings for",
                "in love with",
                "romantic feelings",
            )):
                guidance.append(
                    f"{name}: established private relationship fact involving {person} -> {text_value}"
                )

    return guidance


def build_scene_prompt(data):
    """Build a compact, high-salience scene request for the writer."""
    def clean(value, default=""):
        value = str(value or "").strip()
        return value if value else default

    chapter = clean(data.get("chapter"), "1")
    scene = clean(data.get("scene"), "1")
    characters = clean(data.get("characters"))
    goal = clean(data.get("goal"))
    required = clean(data.get("required"))
    avoid = clean(data.get("avoid"))
    tone = clean(data.get("tone"))
    length = clean(data.get("length"), "500 to 650")
    guidance = clean(data.get("guidance"))

    if not goal:
        raise ValueError("Scene goal is required.")

    revelation_guidance = _derive_revelation_guidance(characters)

    lines = [
        f"Begin Chapter {chapter}, Scene {scene}.",
        "",
        f"Write a complete scene of approximately {length} words.",
        "Do not stop after the opening setup or a few exchanges.",
        "",
        "ONLY THESE CHARACTERS ARE IN THIS SCENE:",
        characters or "Use only the characters established in the current scene context.",
        "Do not introduce, place, or focus another named character in the scene just because that character exists in the story files. Other established characters may be mentioned or discussed naturally, but they are not physically present unless listed above.",
        "",
        "SCENE GOAL:",
        goal,
    ]

    if required:
        lines += [
            "",
            "MUST HAPPEN BEFORE THE SCENE ENDS:",
            required,
            "",
            "For this required event, use this exact sequence:",
            "1. Lead naturally into the event.",
            "2. Have the required character actually say the revealing words in dialogue.",
            "3. Make the spoken meaning clear enough that the other character understands what slipped out.",
            "4. Have the speaker immediately realize the mistake.",
            "5. Have the speaker cover it naturally with a correction, joke, denial, or change of subject.",
            "6. Show the other character noticing and reacting.",
            "7. Continue the scene briefly after the reaction.",
            "Do not replace any of these steps with narration, thoughts, glances, or implied meaning.",
            "Do not end the scene until every required step has actually happened.",
        ]

    if revelation_guidance:
        lines += [
            "",
            "ESTABLISHED PRIVATE FACTS RELEVANT TO THE REQUIRED REVELATION:",
            *[
                f"- {item}"
                for item in revelation_guidance
            ],
            "Use a relevant established fact for the required spoken reveal rather than inventing a different secret.",
        ]

    if avoid:
        lines += [
            "",
            "DO NOT ADVANCE PAST THIS BOUNDARY:",
            avoid,
        ]

    if tone:
        lines += ["", "TONE / STYLE:", tone]

    if guidance:
        lines += ["", "CREATIVE GUIDANCE:", guidance]

    lines += [
        "",
        "WRITE NOW.",
        "Start from the actual current story position and previous saved section.",
        "Do not invent an earlier conversation to respond to.",
        "Stay with the listed characters and the current scene.",
        "Ordinary chit-chat and small everyday details are welcome and are temporary scene material.",
        "Do not add a new major plot event, clue, threat, location, relationship, or other important story fact.",
        f"Complete the full scene at approximately {length} words.",
        "End at a natural stopping point without crossing the protected boundary.",
        "Output only the story prose.",
    ]

    return "\n".join(lines).strip()


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


def available_story_names():
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        [path.name for path in STORY_DIR.glob("*.json") if path.is_file()],
        key=str.lower
    )


def load_story_files(names=None):
    names = discover_story_names() if names is None else names

    # These files are mandatory for every story session.
    # Character cards and other reference files remain selectable.
    mandatory = [
        "story_bible.json",
        "current_state.json",
    ]

    names = (
        mandatory
        + [
            name
            for name in names
            if name not in mandatory
        ]
    )

    unknown = [name for name in names if name not in available_story_names()]
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
    return load_story_files(discover_story_names())

CHARACTER_CONTEXT_FIELDS = (
    "name",
    "age",
    "appearance",
    "personality",
    "relationships",
    "background",
    "skills",
    "strengths",
    "weaknesses",
    "important_items",
    "knowledge_rule",
)

CURRENT_STATE_CONTEXT_FIELDS = (
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
)

STORY_BIBLE_CONTEXT_FIELDS = (
    "title",
    "version",
    "status",
    "relationships",
    "locations",
)


def _compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def _pick_fields(data, fields):
    return {
        key: data[key]
        for key in fields
        if key in data
    }


def _format_locked_state(data):
    lines = [
        "[CURRENT SCENE]",
        f"Chapter {data.get('chapter')} / Scene {data.get('scene')}",
        f"Status: {data.get('status')}",
    ]

    location = data.get("location")
    if isinstance(location, dict):
        lines.append("")
        lines.append("WHERE EVERYONE IS:")
        for character, place in location.items():
            lines.append(f"- {character}: {place}")
    elif location:
        lines.extend(["", f"LOCATION: {location}"])

    situation = data.get("current_situation")
    if situation:
        lines.extend(["", "RIGHT NOW:", str(situation)])

    knowledge = data.get("character_knowledge")
    if knowledge:
        lines.append("")
        lines.append("CHARACTER KNOWLEDGE:")
        if isinstance(knowledge, dict):
            for character, facts in knowledge.items():
                if isinstance(facts, list):
                    lines.append(f"- {character}: " + "; ".join(str(f) for f in facts))
                else:
                    lines.append(f"- {character}: {facts}")

    objectives = data.get("active_objectives")
    if objectives:
        lines.append("")
        lines.append("IMMEDIATE OBJECTIVES:")
        if isinstance(objectives, dict):
            for character, objective in objectives.items():
                lines.append(f"- {character}: {objective}")
        else:
            lines.append(str(objectives))

    clues = data.get("active_clues")
    if clues:
        lines.append("")
        lines.append("ESTABLISHED CLUES:")
        for item in clues if isinstance(clues, list) else [clues]:
            lines.append(f"- {item}")

    lines.extend([
        "",
        "Use this as the exact present situation, not as narration to repeat in the prose.",
        "Only the present facts above are writer context. Future plot information is intentionally omitted.",
        "Ordinary temporary movement, sensory detail, body language, and casual dialogue are allowed.",
        "Do not invent material facts merely because they are plausible.",
        "Do not manufacture a new event simply because the scene needs more words.",
        "",
        "END CURRENT SCENE",
    ])

    return "\n".join(lines)


def build_writer_scene_packet(files, scene_characters=None):
    """Build focused present-story context for the prose writer."""
    parsed = {}

    for name, raw in files:
        try:
            parsed[name] = json.loads(raw)
        except Exception:
            continue

    state = parsed.get("current_state.json", {})
    bible = parsed.get("story_bible.json", {})

    character_files = [
        str(name).strip()
        for name in bible.get("character_cards", [])
        if str(name).strip()
    ]

    available_names = []

    for card_name in character_files:
        data = parsed.get(card_name)
        if not isinstance(data, dict):
            continue

        name = str(data.get("name", "")).strip()
        if name:
            available_names.append(name)

    if scene_characters:
        requested = [str(name).strip() for name in scene_characters if str(name).strip()]
        active_names = [
            name
            for name in requested
            if name in available_names
        ]
    else:
        active_names = list(available_names)

    location = state.get("location")

    lines = [
        "[STORY CONTEXT]",
        "",
        "[SCENE CAST]",
        "Only these characters are physically present in this scene:",
        "- " + (", ".join(active_names) if active_names else "Use the characters explicitly listed by the scene request."),
        "Other characters may exist elsewhere in the story and may be mentioned, but they are not physically present unless listed above.",
        f"Chapter: {state.get('chapter')}",
        f"Scene: {state.get('scene')}",
    ]

    if state.get("time"):
        lines.append(f"Time: {json.dumps(state.get('time'), ensure_ascii=False)}")

    if isinstance(location, dict):
        lines.append("")
        lines.append("CURRENT LOCATIONS FOR SCENE CAST:")
        for name, place in location.items():
            if name in active_names:
                lines.append(f"- {name}: {place}")
    elif location:
        lines.extend(["", f"CURRENT LOCATION: {location}"])

    if state.get("current_situation"):
        lines.extend(["", "CURRENT SITUATION:", str(state.get("current_situation"))])

    knowledge = state.get("character_knowledge")
    if isinstance(knowledge, dict):
        relevant = []
        for name in active_names:
            facts = knowledge.get(name)
            if isinstance(facts, list) and facts:
                relevant.append(f"- {name}: " + "; ".join(str(x) for x in facts))
            elif facts:
                relevant.append(f"- {name}: {facts}")
        if relevant:
            lines.extend(["", "WHAT THEY KNOW:"] + relevant)

    objectives = state.get("active_objectives")
    if isinstance(objectives, dict):
        relevant = []
        for name in active_names:
            value = objectives.get(name)
            if value:
                relevant.append(f"- {name}: {value}")
        if relevant:
            lines.extend(["", "CURRENT OBJECTIVES:"] + relevant)

    for card_name in character_files:
        data = parsed.get(card_name)
        if not isinstance(data, dict):
            continue

        name = data.get("name")
        if not name or (active_names and name not in active_names):
            continue

        background = str(data.get("background", "")).strip()
        if any(term in background.lower() for term in ("died", "deceased", "passed away")):
            lines.extend([
                "",
                f"HARD CHARACTER FACT: {name}",
                background,
                "Do not contradict this fact or portray the deceased person as living or present.",
            ])

        filtered = _pick_fields(
            data,
            (
                "name",
                "age",
                "appearance",
                "personality",
                "relationships",
                "background",
                "skills",
                "strengths",
                "weaknesses",
                "important_items",
            )
        )

        lines.extend([
            "",
            f"CHARACTER: {name}",
            _compact_json(filtered),
        ])

    continuity = state.get("continuity_requirements")
    if isinstance(continuity, list) and continuity:
        lines.extend([
            "",
            "PERSISTENT CONTINUITY RULES:",
        ])
        for item in continuity:
            lines.append(f"- {item}")

    completed = state.get("completed_events")
    if completed:
        lines.append("")
        lines.append("IMPORTANT ESTABLISHED EVENTS:")
        for item in completed:
            lines.append(f"- {item}")

    clues = state.get("active_clues")
    if clues:
        lines.append("")
        lines.append("ACTIVE STORY CLUES:")
        for item in clues:
            lines.append(f"- {item}")

    lines.extend([
        "",
        "[END STORY CONTEXT]",
    ])

    return "\n".join(lines)


def build_runtime_story_context(files):
    sections = []

    # Character files are declared by the active story_bible.json.
    # The application does not know character names in advance.
    character_files = set()

    for ref_name, ref_content in files:
        if ref_name != "story_bible.json":
            continue

        try:
            ref_data = json.loads(ref_content)
        except Exception:
            ref_data = {}

        cards = ref_data.get("character_cards", [])

        if isinstance(cards, list):
            character_files = {
                str(item).strip()
                for item in cards
                if str(item).strip()
            }

        break

    for name, content in files:
        try:
            data = json.loads(content)
        except Exception:
            sections.append(
                f"[Reference File: {name}]\n{content}"
            )
            continue

        if name in character_files:
            filtered = _pick_fields(
                data,
                CHARACTER_CONTEXT_FIELDS
            )

            sections.append(
                f"[Runtime Reference: {name}]\n"
                + _compact_json(filtered)
            )

        elif name == "current_state.json":
            filtered = _pick_fields(
                data,
                CURRENT_STATE_CONTEXT_FIELDS
            )

            knowledge = filtered.get("character_knowledge")

            if isinstance(knowledge, dict):
                compact_knowledge = {}

                for character, facts in knowledge.items():
                    if isinstance(facts, list):
                        compact_knowledge[character] = facts[-8:]
                    else:
                        compact_knowledge[character] = facts

                filtered["character_knowledge"] = compact_knowledge

            sections.append(
                _format_locked_state(filtered)
            )

        elif name == "story_bible.json":
            filtered = _pick_fields(
                data,
                STORY_BIBLE_CONTEXT_FIELDS
            )

            sections.append(
                f"[Runtime Reference: {name}]\n"
                + _compact_json(filtered)
            )

        else:
            sections.append(
                f"[Runtime Reference: {name}]\n"
                + _compact_json(data)
            )

    if not sections:
        return ""

    return (
        "\nRUNTIME STORY CONTEXT START\n"
        + "\n\n".join(sections)
        + "\nRUNTIME STORY CONTEXT END\n"
    )


def build_scene_anchor(files):
    # Runtime story context is already supplied in the system prompt.
    # Repeating current_state here encourages the writer to narrate the
    # state instead of writing the scene.
    return ""


def build_previous_section_context(max_chars=12000):
    """Load the latest saved manuscript section as a continuity bridge.

    The writer process intentionally clears llama.cpp's conversation before
    each request, so it cannot remember the last generated prose by itself.
    The latest saved manuscript section provides the missing prose continuity
    without relying on browser chat history.

    This text is continuity context only. Character files, story_bible.json,
    current_state.json, and the user's current request remain authoritative.
    """
    try:
        state = read_current_state()
    except Exception:
        return ""

    chapter = state.get("chapter")

    if not isinstance(chapter, int):
        return ""

    chapter_dir = (
        MANUSCRIPT_DIR
        / "Chapters"
        / f"Chapter_{chapter:02d}"
    )

    if not chapter_dir.is_dir():
        return ""

    candidates = []

    for path in chapter_dir.glob(
        f"Chapter_{chapter:02d}_Section_*.txt"
    ):
        match = re.search(
            r"_Section_(\d+)\.txt$",
            path.name
        )

        if not match:
            continue

        try:
            number = int(match.group(1))
        except ValueError:
            continue

        candidates.append((number, path))

    if not candidates:
        return ""

    if not candidates:
        return ""

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, previous_path = candidates[0]

    try:
        previous_text = previous_path.read_text(
            encoding="utf-8"
        ).strip()
    except Exception:
        return ""

    if not previous_text:
        return ""

    if len(previous_text) > max_chars:
        previous_text = previous_text[-max_chars:]

    return (
        "\n[PREVIOUS SAVED STORY SECTION]\n"
        "Continue directly from the end of this section. "
        "Do not restart it or recap it. "
        "Use it only as a continuity bridge. "
        "It does not override authoritative story context.\n\n"
        + previous_text
        + "\n[END PREVIOUS SAVED STORY SECTION]\n"
    )


class LlamaSession:
    def __init__(self):
        self.child = None
        self.lock = threading.RLock()
        self.system_prompt = SYSTEM_PROMPT
        self.files = []
        self.story_dir = STORY_DIR
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

        cast_match = re.search(
            r"ONLY THESE CHARACTERS ARE IN THIS SCENE:\s*([^\n]+)",
            base_prompt,
            re.IGNORECASE,
        )

        scene_characters = []
        if cast_match:
            scene_characters = [
                item.strip()
                for item in cast_match.group(1).split(",")
                if item.strip()
            ]

        scene_packet = build_writer_scene_packet(
            files,
            scene_characters=scene_characters,
        )

        if scene_packet:
            parts.append(scene_packet)

        previous_section = build_previous_section_context()

        if previous_section:
            parts.append(previous_section)

        return "\n".join(parts)

    def _read_more(self, timeout):
        # Retained for compatibility with older code paths. Writer generation
        # no longer depends on interactive prompt detection.
        if self.child is None or self.child.stdout is None:
            raise RuntimeError("llama-cli stdout is unavailable")

        import select

        ready, _, _ = select.select(
            [self.child.stdout],
            [],
            [],
            timeout
        )

        if not ready:
            return ""

        chunk = os.read(
            self.child.stdout.fileno(),
            8192
        )

        if not chunk:
            return ""

        return chunk.decode(
            "utf-8",
            errors="replace"
        )

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
                [self.child.stdout], [], [], timeout
            )

            if not ready:
                break

            chunk = os.read(self.child.stdout.fileno(), 8192)

            if not chunk:
                break

    def _wait_for_prompt(self, timeout, initial=False):
        deadline = time.time() + timeout
        text = self.buffer

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
                min(1.0, max(0.05, deadline - time.time()))
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
        raw = (text or "").replace("\r", "")

        # llama-cli may echo its banner and the complete prompt even when
        # --no-display-prompt is supplied. Strip that transport output before
        # the story reaches the chat UI or validation.
        if "USER REQUEST:" in raw:
            raw = raw.rsplit("USER REQUEST:", 1)[1]

        for marker in (
            "[END PREVIOUS SAVED STORY SECTION]",
            "[END SCENE PACKET]",
        ):
            if marker in raw:
                raw = raw.rsplit(marker, 1)[1]

        lines = []

        for line in raw.splitlines():
            s = line.strip()

            if not s:
                if lines and lines[-1] != "":
                    lines.append("")
                continue

            if s.startswith("[ Prompt:") or s.startswith("[ Generation:"):
                continue

            if s.startswith("Exiting..."):
                continue

            if s == ">":
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
            self.started_at = time.time()
            self.buffer = ""

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
                "-t", "4",
                "-c", "8192",
                "--temp", "0.65",
                "--reasoning", "off",
                "--repeat-last-n", "256",
                "--repeat-penalty", "1.08",
                "--n-predict", "1800",
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

    def ask(self, text, clear_history=True):
        with self.lock:
            proc = self.child

            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    "Session is not running. Click New Chat to start it."
                )

            started = time.time()

            try:
                if clear_history:
                    proc.stdin.write(
                        b"/clear\n"
                    )
                    proc.stdin.flush()

                    self._wait_for_prompt(
                        30,
                        initial=False
                    )

                    self.buffer = ""
                    self._drain_pending_output(
                        quiet_window=0.5
                    )

                proc.stdin.write(
                    ("USER REQUEST:\n" + text + "\n").encode("utf-8")
                )
                proc.stdin.flush()

                raw = self._wait_for_prompt(
                    300,
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
        rendered_prompt = self._build_prompt(
            self.system_prompt,
            self.files
        )

        total = len(rendered_prompt) + len(extra)

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
                "running": bool(session.child and session.child.poll() is None),
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
                        "exists": (STORY_DIR / name).exists(),
                            "default": name in discover_story_names(),
                    }
                    for name in available_story_names()
                ],
                "available_models": available_models(),
                "story_dir": str(STORY_DIR),
                "models_dir": str(MODELS_DIR),
                "context_size": 8192,
                "gpu_layers": 0,
                "reasoning": "off",
                "n_predict": 2400,
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
                    "next_save_section": (
                        next_story_section_path(
                            read_current_state().get("chapter")
                        )[0]
                        if STATE_PATH.exists()
                        and isinstance(read_current_state().get("chapter"), int)
                        and isinstance(read_current_state().get("scene"), int)
                        else None
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

        if path == "/api/stories":
            self._json(
                200,
                {
                    "stories": [
                        {
                            "name": story.name,
                            "active": story.resolve() == STORY_DIR.resolve()
                        }
                        for story in available_story_dirs()
                    ],
                    "active": STORY_DIR.name,
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
                            "default": name in discover_story_names(),
                            "loaded": any(
                                loaded_name == name
                                for loaded_name, _ in session.files
                            )
                        }
                        for name in available_story_names()
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

            if path == "/api/story/select":
                name = str(body.get("name", "")).strip()

                if not name:
                    raise ValueError("No story was selected.")

                story_dir = STORIES_ROOT / name
                set_active_story_dir(story_dir)

                session.stop()
                session.files = load_story_files(
                    discover_story_names()
                )
                session.story_dir = STORY_DIR
                pending_state = None

                self._json(
                    200,
                    {
                        "ok": True,
                        "story": STORY_DIR.name,
                        "story_dir": str(STORY_DIR),
                        "files": discover_story_names()
                    }
                )

                return

            if path == "/api/scene-prompt":
                prompt = build_scene_prompt(body)

                state = read_current_state()

                self._json(
                    200,
                    {
                        "ok": True,
                        "prompt": prompt,
                        "chapter": state.get("chapter"),
                        "scene": state.get("scene"),
                        "status": state.get("status"),
                    }
                )

                return

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
                    names = discover_story_names()

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
                session.files = load_story_files(
                    [
                        "story_bible.json",
                        "current_state.json",
                    ]
                )

                self._json(
                    200,
                    {
                        "ok": True,
                        "files": [
                            name
                            for name, _ in session.files
                        ]
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
                        "context_size": 8192
                    }
                )

                return

            if path == "/api/start":
                requested_files = body.get("files")

                if requested_files:
                    names = []

                    for item in requested_files:
                        if isinstance(item, dict):
                            name = str(
                                item.get(
                                    "name",
                                    "unnamed.json"
                                )
                            )[:120]
                        else:
                            name = str(item)[:120]

                        if name and name not in names:
                            names.append(name)

                    files = load_story_files(names)
                else:
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
                        "context_size": 8192
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

                requested_chapter = body.get("chapter")
                requested_scene = body.get("scene")

                try:
                    requested_chapter = (
                        int(requested_chapter)
                        if requested_chapter is not None
                        else None
                    )
                except (TypeError, ValueError):
                    requested_chapter = None

                try:
                    requested_scene = (
                        int(requested_scene)
                        if requested_scene is not None
                        else None
                    )
                except (TypeError, ValueError):
                    requested_scene = None

                proposed = generate_state_proposal(
                    story,
                    body.get("sectionFilename"),
                    requested_chapter,
                    requested_scene,
                )

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

                answer, elapsed = session.ask(text)

                self._json(
                    200,
                    {
                        "ok": True,
                        "content": answer,
                        "seconds": round(elapsed, 1),
                        "estimated_context_tokens": (
                            session.estimate_context(
                                text + "\n" + answer
                            )
                        ),
                        "context_size": 8192
                    }
                )

                return

            if path == "/api/save-story":
                story = str(body.get("story", "")).strip()

                if not story:
                    self._json(
                        400,
                        {
                            "error": "Story text is empty"
                        }
                    )
                    return

                try:
                    state = read_current_state()
                except Exception as exc:
                    self._json(
                        500,
                        {
                            "error": "Could not read current_state.json: " + str(exc)
                        }
                    )
                    return

                chapter = state.get("chapter")
                scene = state.get("scene")

                if not isinstance(chapter, int) or not isinstance(scene, int):
                    self._json(
                        500,
                        {
                            "error": "current_state.json does not contain valid chapter and scene numbers"
                        }
                    )
                    return

                requested_filename = str(
                    body.get("filename", "")
                ).strip()

                overwrite = bool(
                    body.get("overwrite", False)
                )

                if requested_filename:
                    match = re.fullmatch(
                        r"Chapter_(\d+)_Section_(\d+)\.txt",
                        requested_filename,
                        re.IGNORECASE
                    )

                    if not match:
                        raise ValueError(
                            "Invalid manuscript section filename."
                        )

                    target_chapter = int(match.group(1))
                    target_scene = int(match.group(2))

                    if target_chapter != chapter:
                        raise ValueError(
                            "Manuscript section must belong to the current chapter."
                        )

                    if target_scene < scene:
                        raise ValueError(
                            "Manuscript section cannot be earlier than the current state scene."
                        )

                    filename = (
                        f"Chapter_{target_chapter:02d}_Section_{target_scene:02d}.txt"
                    )
                    path_out = (
                        MANUSCRIPT_DIR
                        / "Chapters"
                        / f"Chapter_{target_chapter:02d}"
                        / filename
                    )

                    if path_out.exists() and not overwrite:
                        raise ValueError(
                            f"Manuscript section already exists: {filename}"
                        )
                else:
                    _, filename, path_out = next_story_section_path(
                        chapter
                    )

                with STORY_SAVE_LOCK:
                    path_out.parent.mkdir(parents=True, exist_ok=True)

                    if path_out.exists() and not overwrite:
                        raise ValueError(
                            f"Manuscript section already exists: {filename}"
                        )

                    path_out.write_text(
                        story + "\n",
                        encoding="utf-8"
                    )

                self._json(
                    200,
                    {
                        "ok": True,
                        "filename": filename,
                        "path": str(path_out)
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
