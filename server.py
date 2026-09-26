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

SYSTEM_PROMPT = r'''You are the prose-writing engine for an ongoing fictional story.

Use the supplied story context as private reference material. Write the user's requested scene as finished fiction.

Use established character facts, relationships, locations, timeline, and current state for continuity. Characters should know only what the story context establishes they know. Do not reveal hidden information simply because it exists in reference material.

Follow the user's current scene request, including requested characters, tone, boundaries, and beats.

Write naturally. Let dialogue, action, reactions, and sensory detail carry the scene. Creative everyday conversation and temporary background detail are allowed when they fit the scene.

Do not add a new major plot event, permanent character fact, relationship, clue, identity, location, or backstory unless the story context or the user's request establishes it.

When a previous saved section is supplied, continue from its ending instead of restarting or recapping it.

Output only the story prose.'''


'''


def generate_state_proposal(story_text, section_filename=None):
    global pending_state

    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    current_state = read_current_state()

    section_chapter = None
    section_scene = None

    if section_filename:
        match = re.search(
            r"^Chapter_(\d+)_Section_(\d+)\.txt$",
            str(section_filename).strip(),
            re.IGNORECASE
        )

        if match:
            section_chapter = int(match.group(1))
            section_scene = int(match.group(2))

    prompt = f"""Identify ONLY the changes caused by this completed story section.

The patch may contain ONLY these top-level fields:
status, chapter, scene, scene_completed, location, time,
current_situation, character_knowledge, completed_events, active_clues,
new_clues, unresolved_questions, active_objectives, continuity_requirements.

Never invent or create any other top-level field.

CURRENT STATE BEFORE THIS SECTION:
{json.dumps(current_state, ensure_ascii=False, indent=2)}

COMPLETED STORY SECTION:
{story_text}

Use CURRENT STATE BEFORE THIS SECTION as the baseline for comparison.
Only propose information that is genuinely new or changed because of the completed story section.
If a fact already exists in current state, do NOT include it again.
Do not treat ordinary restatement, repeated context, or pre-existing knowledge as a change.

Return ONLY this JSON structure:

{{
  "patch": {{}},
  "evidence": []
}}

Rules:
- "patch" contains ONLY information newly established or changed by this story section relative to CURRENT STATE BEFORE THIS SECTION.
- Use the supplied current state as the comparison baseline. Do NOT treat existing facts as new.
- Do NOT copy unchanged information from current state.
- Do NOT invent information.
- Do NOT include unchanged information.
- For array fields, include ONLY new items introduced by this section.
- Do NOT include old array items.
- The ONLY permitted top-level patch fields are:
  status, chapter, scene, scene_completed, location, time, current_situation,
  character_knowledge, completed_events, active_clues, new_clues,
  unresolved_questions, active_objectives, continuity_requirements.
- Do NOT create or return any other top-level fields. Fields such as
  conversation_topics, current_activity, notes, summary, history, or metadata are invalid.
- "evidence" must contain an exact quote from this story section for every substantive patch item.
- Every evidence "quote" MUST be one contiguous excerpt copied character-for-character from the completed story section after normal whitespace cleanup.
- Do NOT use ellipses ("..."), brackets, summaries, stitched excerpts, or text from CURRENT STATE BEFORE THIS SECTION.
- Prefer one short complete sentence or one short contiguous sentence fragment as the quote.
- The quote must directly support the claim. Do not attach a baseline-only fact to a quote that merely provides nearby context.
- If a claim cannot be supported by a direct contiguous quote from the completed story section, do NOT include that claim in the patch.
- Never create an update merely because a fact from current state remains true. Existing facts are not changes.
- Treat location and physical position as literal continuity data. Do not infer arrival at a place from language such as approaching, nearing, heading toward, or not yet reached.
- When a character's physical position changes, update current_situation as needed so it remains consistent with the changed location. Do not leave current_situation describing an earlier physical position when location has advanced.
- Do not make a character appear to be at or passing a location unless the completed story section explicitly establishes that position.
- Do not create active_clues or unresolved_questions from ordinary objects, dialogue, or curiosity unless the story section explicitly establishes them as plot-relevant clues or unresolved story questions.
- Do not add temporary observations, gestures, glances, blushes, emotions, or ordinary sensory details to character_knowledge unless the section establishes a meaningful new fact that the character learned and may need to remember later.
- Do not add already-established character traits, possessions, relationships, or background facts to current state merely because the section mentions or shows them.
- A temporary observation about an already-established person, possession, relationship, or setting is NOT a new character-knowledge fact. For example, noticing, glancing at, touching, carrying, or mentioning an established object does not create persistent knowledge.
- Character knowledge should be updated only when the section gives the character a genuinely new fact, discovery, instruction, confession, witness account, or other information that can matter after the immediate scene.
- Do not use character_knowledge as a log of moment-to-moment perception. Do not record ordinary noticing, looking, remembering an established fact, or wondering about something unless it creates a meaningful new piece of knowledge.
- Do not add a continuity_requirement for a one-time action, observation, or ordinary piece of scene texture. Continuity requirements are only for facts or constraints that must remain true in later scenes.
- In particular, a character noticing an established object is not a state change unless the noticing itself creates a meaningful new plot or knowledge consequence.
- If nothing changed, return exactly:
  {{"patch": {{}}, "evidence": []}}

Do not return markdown or explanations.
Do not return current_state.json.
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
            "--n-predict", "900",
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

        if section_chapter is not None and section_scene is not None:
            if section_chapter < current_state.get("chapter", section_chapter):
                raise ValueError(
                    "Selected manuscript section is older than the current story state."
                )

            patch["chapter"] = section_chapter
            patch["scene"] = section_scene

        patch = _filter_patch_to_supported_evidence(
            current_state,
            patch,
            story_text,
            evidence
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
    length = clean(data.get("length"), "500 to 650")
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

    # Put the final scene instructions immediately before generation.
    # Smaller local models are more reliable when the actual required beat is
    # repeated at the end of the request instead of being buried earlier.
    if narrative_focus_lines:
        lines += [""] + narrative_focus_lines

    if required:
        lines += [
            "",
            "FINAL REQUIREMENTS CHECK - DO THIS ON THE PAGE:",
            "Before ending the scene, silently verify that every REQUIRED item below "
            "actually occurs in the prose.",
            "Do not satisfy a required interaction, reveal, action, or emotional beat "
            "with narration that merely says it happened or implies it. Show it through "
            "observable action, dialogue, or character reaction.",
            "When a REQUIRED beat is an accidental spoken reveal, the character must "
            "actually say the revealing thing in dialogue. The other character must "
            "have an opportunity to hear or react to it.",
            "REQUIRED ITEMS TO FULFILL:",
            required,
        ]

    if avoid:
        lines += [
            "",
            "FINAL BOUNDARY CHECK:",
            "Do not cross or advance any event listed under DO NOT ADVANCE YET.",
            "DO NOT ADVANCE YET:",
            avoid,
        ]

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


def build_writer_scene_packet(files):
    """Build the smallest useful context for the prose writer.

    The writer sees the current scene, active characters, their relevant
    established traits/relationships, current knowledge, objectives, and
    established locations. Future plot rules and hidden canon stay outside
    the normal writing context.
    """
    parsed = {}

    for name, raw in files:
        try:
            parsed[name] = json.loads(raw)
        except Exception:
            continue

    state = parsed.get("current_state.json", {})
    bible = parsed.get("story_bible.json", {})

    character_files = {
        str(name).strip()
        for name in bible.get("character_cards", [])
        if str(name).strip()
    }

    location = state.get("location")
    active_names = []

    if isinstance(location, dict):
        for name in location:
            if any(
                card_data.get("name") == name
                for card_name, card_data in parsed.items()
                if card_name in character_files
                and isinstance(card_data, dict)
            ):
                active_names.append(name)

    if not active_names:
        for card_name in character_files:
            card_data = parsed.get(card_name)
            if isinstance(card_data, dict) and card_data.get("name"):
                active_names.append(card_data["name"])

    lines = [
        "[SCENE PACKET]",
        f"Chapter: {state.get('chapter')}",
        f"Scene: {state.get('scene')}",
    ]

    time_data = state.get("time")
    if time_data:
        lines.append(f"Time: {json.dumps(time_data, ensure_ascii=False)}")

    if isinstance(location, dict):
        lines.append("")
        lines.append("WHERE:")
        for name, place in location.items():
            lines.append(f"- {name}: {place}")
    elif location:
        lines.extend(["", f"WHERE: {location}"])

    situation = state.get("current_situation")
    if situation:
        lines.extend(["", "RIGHT NOW:", str(situation)])

    knowledge = state.get("character_knowledge")
    if isinstance(knowledge, dict):
        relevant_knowledge = []
        for name in active_names:
            facts = knowledge.get(name)
            if isinstance(facts, list):
                relevant_knowledge.append(
                    f"- {name}: " + "; ".join(str(f) for f in facts)
                )
            elif facts:
                relevant_knowledge.append(
                    f"- {name}: {facts}"
                )

        if relevant_knowledge:
            lines.extend(["", "WHAT THEY KNOW:"] + relevant_knowledge)

    objectives = state.get("active_objectives")
    if isinstance(objectives, dict):
        relevant_objectives = []
        for name in active_names:
            objective = objectives.get(name)
            if objective:
                relevant_objectives.append(
                    f"- {name}: {objective}"
                )

        if relevant_objectives:
            lines.extend(["", "WHAT THEY ARE DOING:"] + relevant_objectives)

    character_fields = (
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

    for card_name in character_files:
        data = parsed.get(card_name)
        if not isinstance(data, dict):
            continue

        name = data.get("name")
        if name not in active_names:
            continue

        filtered = _pick_fields(data, character_fields)

        lines.extend([
            "",
            f"CHARACTER: {name}",
            _compact_json(filtered),
        ])

    locations = bible.get("locations")
    if isinstance(locations, dict) and locations:
        lines.append("")
        lines.append("ESTABLISHED LOCATIONS:")
        for name, description in locations.items():
            lines.append(f"- {name}: {description}")

    completed = state.get("completed_events")
    if completed:
        lines.append("")
        lines.append("ESTABLISHED EVENTS:")
        for item in completed:
            lines.append(f"- {item}")

    clues = state.get("active_clues")
    if clues:
        lines.append("")
        lines.append("ESTABLISHED CLUES:")
        for item in clues:
            lines.append(f"- {item}")

    lines.extend([
        "",
        "Write from this packet. Do not narrate the packet itself.",
        "The packet describes the present scene, not future story information.",
        "Ordinary conversation and temporary sensory detail may be creative.",
        "Do not invent material plot facts, hidden information, or persistent canon.",
        "Do not make characters search for danger without a concrete reason.",
        "",
        "[END SCENE PACKET]",
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

        scene_packet = build_writer_scene_packet(files)

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
                "--n-predict", "1200",
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
                "running": True,
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
                    "next_save_scene": (
                        next_story_section_path(
                            read_current_state().get("chapter"),
                            read_current_state().get("scene")
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

                proposed = generate_state_proposal(
                    story,
                    body.get("sectionFilename")
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

                try:
                    answer, elapsed = session.ask(text)
                except Exception:
                    raise

                self._json(200, {
                    "ok": True,
                    "content": answer,
                    "seconds": round(elapsed, 1),
                    "estimated_context_tokens": (
                        session.estimate_context(
                            text + "\n" + answer
                        )
                    ),
                    "context_size": 8192
                })

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

                    if target_scene <= scene:
                        raise ValueError(
                            "Manuscript section must be later than the current state scene."
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
                else:
                    _, filename, path_out = next_story_section_path(
                        chapter, scene
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
