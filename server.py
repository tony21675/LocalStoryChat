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

SYSTEM_PROMPT = r'''You write prose for an ongoing fictional story.

Use the supplied reference context as private canon. Do not quote, summarize, or discuss the files unless the user explicitly asks.

CANON:
- Character files define character identity, personality, relationships, background, and established character facts.
- story_bible.json defines permanent world canon and story rules.
- current_state.json defines the exact present situation, character knowledge, and scene boundary.
- The user's request defines the immediate writing task.
- Established facts are fixed. Unknown information stays unknown.
- Plausibility is not evidence. Do not promote a plausible detail into canon.
- Never reveal hidden information before it is established in the story.
- Do not change who is present, where they are, or what they can perceive.
- During ordinary moments, characters focus on their current activity and each other. Observant or cautious characters do not continuously scan for danger without a concrete reason.
- Do not invent material plot events, clues, evidence, identities, motives, destinations, important objects, backstory, persistent setting facts, or new relationships.
- Do not add unexplained people, animals, vehicles, objects, suspicious activity, or environmental anomalies just to make prose more interesting.

DIALOGUE:
- Normal everyday conversation may be invented.
- School gossip, jokes, teasing, opinions, complaints, and harmless speculation are welcome.
- Keep invented chatter disposable. Do not turn it into important facts, specific past events, secrets, or future setup unless the story establishes them.
- Existing relationship canon controls romantic framing. Maya may be attracted to Tony. Tiffany and Maya are best friends with a close, sister-like platonic bond.

SCENE:
- Start in the immediate present.
- Let the scene advance through dialogue, actions, reactions, small decisions, or reaching an already-established place.
- A scene does not need a new external event to progress.
- Do not repeat the same state, movement, atmosphere, or explanation just to add length.
- Prefer concrete interaction over decorative description.
- Use only temporary sensory detail that does not create new material facts.
- For an ordinary continuation, write only as much as the moment needs, typically about 400 to 800 words. Write longer only when the request calls for it or the scene genuinely requires it.
- End at a natural break or when the requested moment is complete.

Output only the story prose.'''


CANON_REVIEW_SYSTEM_PROMPT = r"""You are a continuity and canon reviewer for an ongoing fictional story.

Review the generated story draft against the supplied runtime story context and the user's request.

Return ONLY one valid JSON object:

{
  "approved": true,
  "violations": []
}

CORE STANDARD:
Protect established canon without making the prose sterile.

Approve ordinary creative prose when it is temporary, incidental, and does not create a material story fact.

Reject a detail when it changes or establishes information the story may need to remember later.

MATERIAL FACTS THAT REQUIRE SUPPORT:
- new plot events
- important actions that affect what happens in the story
- new characters, identities, or relationships
- backstory, memories, history, or prior events
- specific locations, routes, destinations, or spatial relationships
- important objects, possessions, vehicles, or persistent physical conditions
- clues, evidence, witnesses, leads, motives, plans, or intentions
- character knowledge, observations, discoveries, or information
- injuries or other persistent bodily conditions
- time, duration, sequence, or elapsed-time claims
- dialogue that establishes new factual information
- anything that contradicts established canon

CREATIVE PROSE THAT MAY BE APPROVED:
Ordinary temporary actions and sensory texture may be creative when they do not create material canon.

Examples that may be acceptable:
- wiping hands
- shifting position
- looking down or pausing
- ordinary walking movements
- generic light, weather, sound, smell, or mood
- brief non-factual chatter
- small transitional actions that do not affect the plot

Do NOT reject such details merely because they were not explicitly listed in the JSON.

However, reject them when the wording turns them into material facts.

Examples:
- "Tony wiped his hands on a rag." may be acceptable as temporary prose.
- "Tony always kept a rag beside the truck." requires support because it establishes a habit or persistent detail.
- "Maya kicked a pebble." may be acceptable as incidental movement.
- "Maya took the shortcut she always used." requires support because it establishes a route and routine.
- "They chatted as they walked." may be acceptable.
- "They discussed a teacher's assignment due Friday." requires support because it creates specific story information.
- "Tony looked toward the street." may be acceptable.
- "Tony saw Tiffany approaching from the street." requires support because it establishes an observation and spatial relationship.

UNKNOWN INFORMATION:
- Plausibility is not evidence.
- Keep unknown material unknown.
- Never approve a new fact merely because it would make sense.
- Never reveal hidden canon unless the story context or user request establishes the reveal.

KNOWLEDGE AND OBSERVATION:
Characters may know or notice only what is established by context, the user's request, or events occurring in the current scene.
Reject unsupported important discoveries, sightings, overheard information, memories, or deductions presented as facts.

CONTINUITY:
Reject contradictions of established canon.
Reject new material plot facts.
Reject invented backstory, motives, destinations, evidence, clues, witnesses, identities, relationships, or persistent objects or conditions.
Reject premature reveals of hidden or unknown information.

CONSERVATIVE BUT PROPORTIONAL:
Do not reject ordinary temporary prose merely because it is not explicitly in the JSON.

However, be strict about SPECIFICITY. In a sparse scene, reject concrete details that create factual information the story did not establish, including:
- named teachers, classmates, neighbors, parents, or other people
- specific assignments, classes, school events, or academic history
- named streets, parks, businesses, landmarks, routes, or home features
- pets, vehicles, objects, or possessions not established by canon
- specific family routines, prior events, traditions, or plans
- claims about what another person is doing, has done, or normally does
- new sightings, sounds, movements, or environmental events that could become story facts
- internal thoughts that create unsupported history, attraction, suspicion, anticipation, or relationship dynamics
- wording that turns an ordinary moment into an implied threat, mystery, or future setup

Generic temporary texture remains acceptable, such as ordinary walking, a pebble underfoot, generic weather, light, a brief laugh, or disposable nonspecific chatter.

For every violation, return:
{
  "quote": "EXACT QUOTE from the draft",
  "reason": "why the quoted claim creates an unsupported or contradictory material story fact",
  "category": "backstory|character_knowledge|plot|clue|location|relationship|contradiction|object|observation|dialogue|physical_state|time|other"
}

If there are no material continuity violations, return:
{"approved": true, "violations": []}

If there is at least one material continuity violation, return:
{"approved": false, "violations": [...]}

Do not output markdown, explanations, analysis, or code fences.
"""
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

Return ONLY one valid JSON object using exactly this structure:

{
  "patch": {},
  "evidence": []
}

The "patch" must contain ONLY fields whose values are actually different from CURRENT STATE BEFORE THIS SECTION.
The "evidence" array must prove every substantive new or changed claim in the patch.

CRITICAL COMPACTNESS RULES:
- Do NOT copy unchanged fields from current_state.json into patch.
- Do NOT repeat existing array items.
- Do NOT rewrite existing arrays merely because they already exist.
- If a field has no new information, leave it out of patch.
- If nothing changed, the patch MUST be {}.
- The normal response should be a small JSON object, not a copy of current_state.json.
- Never advance chapter or scene merely because the supplied section already belongs to that chapter or scene.

IMPORTANT:
- Include only top-level fields that changed in "patch".
- Do not repeat unchanged fields.
- For changed nested objects, include only changed nested keys.
- For arrays, return ONLY NEW items introduced by the completed story section. Do not copy existing array items. The application will append new items to the existing array.
- Record only events that actually happened in the supplied story section.
- Never invent future events.
- Never turn an unknown fact into a known fact.
- Keep character knowledge limited to what the character could actually know.
- Do not reveal hidden canon.
- Do not invent clues, locations, motives, identities, evidence, relationships, or backstory.
- If nothing changed, return {"patch": {}, "evidence": []}.

Evidence rules:
- Every substantive new or changed claim in "patch" must have a matching entry in "evidence".
- Each evidence entry must use this structure:
  {
    "field": "field.path",
    "claim": "the exact value or array item being added or changed",
    "quote": "an EXACT QUOTE copied from the completed story section"
  }
- "quote" must be copied exactly from the completed story section. Do not paraphrase it.
- The application will check that every quote actually appears in the completed story section.
- For array fields, provide evidence for every newly added item.
- For changed string or nested values, provide evidence for the changed value.
- Controlled bookkeeping fields such as chapter, scene, scene_completed, and status do not require quotation evidence.
- Do not use evidence to justify information that is merely inferred or possible.
- If the story section does not explicitly support a proposed change, do not include that change.
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

    evidence_keys = set()

    for entry in evidence:
        if not isinstance(entry, dict):
            raise ValueError(
                "State proposal rejected. Every evidence entry must be an object."
            )

        field = entry.get("field")
        claim = entry.get("claim")
        quote = entry.get("quote")

        if not isinstance(field, str) or not field.strip():
            raise ValueError(
                "State proposal rejected. Every evidence entry needs a field."
            )

        if "claim" not in entry:
            raise ValueError(
                "State proposal rejected. Every evidence entry needs a claim."
            )

        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(
                "State proposal rejected. Every evidence entry needs a non-empty quote."
            )

        normalized_quote = _normalize_evidence_text(
            quote
        )

        if normalized_quote not in normalized_story:
            raise ValueError(
                "State proposal rejected. Evidence quote was not found "
                "in the completed story section: "
                f"{quote!r}"
            )

        evidence_keys.add(
            (
                field.strip(),
                _claim_key(claim)
            )
        )

    for field, claim in _iter_changed_claims(base, patch):
        top_level = field.split(".", 1)[0]

        if top_level in _STATE_METADATA_FIELDS:
            continue

        key = (
            field,
            _claim_key(claim)
        )

        if key not in evidence_keys:
            raise ValueError(
                "State proposal rejected. Missing exact evidence for "
                f"changed field '{field}': {claim!r}"
            )


def extract_json_object(text):
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

    # The llama-cli output may echo the prompt before the model response.
    # Prefer the actual state-manager proposal over JSON copied from the prompt.
    for obj in reversed(candidates):
        if "patch" in obj and "evidence" in obj:
            return obj

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


def review_story_draft(user_request, draft):
    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    runtime_context = build_runtime_story_context(
        session.files
    )

    prompt = f"""Review the following generated story draft for canon and continuity.

USER REQUEST:
{user_request}

RUNTIME STORY CONTEXT:
{runtime_context}

GENERATED STORY DRAFT:
{draft}

Return ONLY the required JSON object.
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
        "-c", "8192",
        "--reasoning", "off",
        "--temp", "0.02",
        "--top-k", "20",
        "--top-p", "0.80",
        "--repeat-last-n", "256",
        "--repeat-penalty", "1.08",
        "--n-predict", "700",
        "--system-prompt", CANON_REVIEW_SYSTEM_PROMPT,
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
            "Canon review generation timed out."
        )

    if proc.returncode not in (0, None):
        raise RuntimeError(
            f"Canon reviewer exited with code {proc.returncode}.\\n"
            f"{output[-2000:]}"
        )

    result = extract_json_with_keys(
        output,
        {"approved", "violations"}
    )

    if not isinstance(
        result.get("approved"),
        bool
    ):
        raise ValueError(
            "Canon reviewer returned an invalid 'approved' value."
        )

    violations = result.get("violations")

    if not isinstance(violations, list):
        raise ValueError(
            "Canon reviewer returned an invalid 'violations' array."
        )

    return result


def generate_state_proposal(story_text):
    global pending_state

    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    current_state = read_current_state()

    prompt = f"""Identify ONLY the changes caused by this completed story section.

COMPLETED STORY SECTION:
{story_text}

Return ONLY this JSON structure:

{{
  "patch": {{}},
  "evidence": []
}}

Rules:
- "patch" contains ONLY information newly established by this story section.
- Do NOT copy information from any previous state.
- Do NOT invent information.
- Do NOT include unchanged information.
- For array fields, include ONLY new items introduced by this section.
- Do NOT include old array items.
- "evidence" must contain an exact quote from this story section for every substantive patch item.
- If nothing changed, return exactly:
  {{"patch": {{}}, "evidence": []}}

Do not return markdown or explanations.
Do not return current_state.json.
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


    print("\n--- RAW STATE MANAGER OUTPUT ---", flush=True)
    print(output, flush=True)
    print("--- END RAW STATE MANAGER OUTPUT ---\n", flush=True)

    proposal = extract_json_object(output)

    if not isinstance(proposal, dict):
        raise ValueError(
            "The state manager did not return a JSON object."
        )

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

    # Prevent the model from accidentally creating character fields
    # at the top level of current_state.json.
    current_characters = current_state.get("character_knowledge", {})

    if isinstance(current_characters, dict):
        for misplaced in current_characters:
            patch.pop(misplaced, None)

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

    # new_clues describes only clues introduced by THIS section.
    # Never carry old section clues forward as "new".
    proposed["new_clues"] = patch.get("new_clues", [])

    validate_state(proposed)

    pending_state = proposed

    return proposed


def build_scene_prompt(data):
    """Build a compact, human-editable scene prompt from the UI fields.

    This deliberately does NOT paste the JSON reference files or current_state
    into the prompt. The model already receives those as authoritative context.
    """
    def clean(value, default=""):
        return str(value or "").strip()

    chapter = clean(data.get("chapter"), "1")
    scene = clean(data.get("scene"), "1")
    characters = clean(data.get("characters"))
    goal = clean(data.get("goal"))
    required = clean(data.get("required"))
    avoid = clean(data.get("avoid"))
    tone = clean(data.get("tone"))
    length = clean(data.get("length"), "400 to 800")
    guidance = clean(data.get("guidance"))

    if not goal:
        raise ValueError("Scene goal is required.")

    lines = [
        f"Begin Chapter {chapter}, Scene {scene}.",
        "",
        f"Write approximately {length} words.",
        "",
        "SCENE GOAL:",
        goal,
    ]

    narrative_focus_lines = []

    if characters:
        lines += ["", "CHARACTERS:", characters]

        # Character order is intentional. The first two listed characters
        # are treated as the primary narrative focus; remaining characters
        # are supporting/background unless the scene goal clearly requires
        # otherwise.
        character_list = [
            item.strip()
            for item in characters.replace("\n", ",").split(",")
            if item.strip()
        ]

        if character_list:
            primary = character_list[:2]
            secondary = character_list[2:]

            narrative_focus_lines = [
                "SCENE FOCUS - READ THIS BEFORE WRITING:",
                "The primary narrative focus is: " + ", ".join(primary) + ".",
                "Begin the scene with the primary characters, not a secondary character.",
                "Spend the clear majority of the scene on the primary characters' "
                "actions, dialogue, interaction, and immediate experience.",
                "Do not divide narrative attention evenly among all listed characters.",
            ]

            if secondary:
                narrative_focus_lines.append(
                    "Secondary/background characters: "
                    + ", ".join(secondary) + "."
                )
                narrative_focus_lines.append(
                    "Keep secondary/background characters brief and incidental. "
                    "Do not give them detailed work, extended internal focus, "
                    "or long descriptive passages unless the scene goal explicitly requires it."
                )

    if required:
        lines += ["", "REQUIRED:", required]

    if avoid:
        lines += ["", "DO NOT ADVANCE YET:", avoid]

    if tone:
        lines += ["", "TONE / STYLE:", tone]

    if guidance:
        lines += ["", "CREATIVE GUIDANCE:", guidance]

    lines += [
        "",
        "CONTINUITY:",
        "Start from the exact current situation in current_state.json and respect the authoritative character cards and story_bible.json.",
        "Preserve established relationships, knowledge, locations, timeline, and canon.",
        "Do not reveal hidden information merely because it exists in the reference files.",
        "Do not invent major plot facts or advance events that are explicitly being held back.",
        "Let the scene breathe. Use natural action, dialogue, sensory detail, personality, and small ordinary details where appropriate.",
    ]

    # Put scene focus at the end so the model sees it immediately before writing.
    if narrative_focus_lines:
        lines += [""] + narrative_focus_lines

    lines += [
        "",
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

        runtime_context = build_runtime_story_context(files)

        if runtime_context:
            parts.append(runtime_context)

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
                "-t", "4",
                "-c", "8192",
                "--temp", "0.65",
                "--reasoning", "off",
                "--repeat-last-n", "256",
                "--repeat-penalty", "1.08",
                "--n-predict", "1400",
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
                # Clear llama.cpp's conversation history without unloading
                # the model or rebuilding the model process.
                proc.stdin.write(
                    b"/clear\n"
                )
                proc.stdin.flush()

                self._wait_for_prompt(
                    30,
                    initial=False
                )

                proc.stdin.write(
                    ("USER REQUEST:\n" + text + "\n").encode("utf-8")
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

                # Give short story-writing responses one chance to continue
                # naturally. The same llama-cli session is kept, so the model
                # can continue from exactly where it stopped.
                story_request_markers = (
                    "write",
                    "begin",
                    "start",
                    "continue",
                    "scene",
                    "chapter",
                    "story",
                    "prose",
                )

                request_lower = text.lower()

                is_story_request = any(
                    marker in request_lower
                    for marker in story_request_markers
                )

                # Review the first complete generation directly.
                # Do not automatically extend short scenes before review.
                # Extra generation encourages the model to invent filler.
                if is_story_request:
                    review = review_story_draft(
                        text,
                        answer
                    )

                    if not review.get("approved", True):
                        violations = review.get("violations", [])

                        repair_prompt = (
                            "Rewrite the story draft below to remove ONLY the "
                            "unsupported or contradictory details identified "
                            "by the canon review. Preserve everything that is "
                            "already valid, including natural dialogue, the "
                            "established relationships, the current location, "
                            "and the present scene. Do not replace removed "
                            "details with new names, people, places, objects, "
                            "events, clues, threats, or other invented facts. "
                            "Do not add suspense or foreshadowing. Keep the "
                            "scene simple and natural. Output only the "
                            "corrected story prose.\n\n"
                            "CANON REVIEW VIOLATIONS:\n"
                            + json.dumps(
                                violations,
                                ensure_ascii=False,
                                indent=2
                            )
                            + "\n\n"
                            "DRAFT TO CORRECT:\n"
                            + answer
                        )

                        proc.stdin.write(
                            (repair_prompt + "\n").encode("utf-8")
                        )
                        proc.stdin.flush()

                        repaired_raw = self._wait_for_prompt(
                            900,
                            initial=False
                        )

                        repaired = self.clean_output(
                            repaired_raw
                        )

                        if repaired:
                            repaired_review = review_story_draft(
                                text,
                                repaired
                            )

                            if repaired_review.get("approved", False):
                                answer = repaired
                            else:
                                print(
                                    "\n--- CANON REVIEW REMAINED UNRESOLVED ---",
                                    flush=True
                                )
                                print(
                                    json.dumps(
                                        repaired_review,
                                        ensure_ascii=False,
                                        indent=2
                                    ),
                                    flush=True
                                )
                                print(
                                    "--- END CANON REVIEW ---\n",
                                    flush=True
                                )
                                answer = repaired

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

                chapter_dir = MANUSCRIPT_DIR / "Chapters" / f"Chapter_{chapter:02d}"
                chapter_dir.mkdir(parents=True, exist_ok=True)

                filename = f"Chapter_{chapter:02d}_Section_{scene:02d}.txt"
                path_out = chapter_dir / filename

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
