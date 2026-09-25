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
- In a sparse ordinary scene, keep background detail generic unless the context establishes the specific thing. Prefer generic light, weather, pavement, breeze, distant sound, or ordinary movement over introducing a specific dog, named person, vehicle, appliance, neighborhood activity, or other concrete background fact.
- When a REQUIRED beat says one character has feelings for a specific person, keep the target unambiguous. The hint may be subtle, but it must clearly point to the requested person by name or an unmistakable relationship reference such as "your dad" or "your father". Do not redirect the hint toward the conversation partner or another character.
- Scene character boundaries matter: use the characters explicitly requested for the scene as the active cast. Do not introduce, speak for, or give narrative focus to another established character merely because that character exists in the reference files. A different character may appear only when the current state, previous section, or user's request establishes that character's presence.

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
- When a previous saved section is supplied, continue directly from its ending instead of restarting or recapping it.
- Treat explicit scene requirements in the user's request as mandatory. Before finishing, silently verify every item under REQUIRED is fulfilled in the prose, even when the requested beat is subtle.
- Treat DO NOT ADVANCE YET as a hard scene boundary. Do not advance, reveal, or invent any listed event.
- Physical continuity is monotonic: if the previous section establishes that a character has passed a location or reached a point on the route, the next scene starts from that position. Never move characters backward to an earlier location unless the user's current request explicitly requires it.
- The previous saved section is a continuity bridge only. It does not override character files, story_bible.json, current_state.json, or the user's current request.
- For an ordinary continuation, write roughly 400 to 650 words unless the user's request specifies a different range. Treat the requested range as guidance, not a reason to pad the scene artificially.
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

TASK REQUIREMENTS:
- Treat the user's SCENE GOAL, REQUIRED items, DO NOT ADVANCE YET items, requested character list, and requested word count as actual requirements.
- Reject a draft when it clearly omits a required character beat, reaction, interaction, or event.
- Reject a draft when it clearly advances an explicitly prohibited event.
- Reject a draft when physical movement contradicts current state or the ending position of the previous saved section.
- A subtle requested beat still counts as required. Mentioning the topic without performing the requested action or reaction is not enough.

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

IMPORTANT DECISION RULE:
When deciding whether a detail is disposable or material, require support from the runtime context for concrete specificity.
A named teacher, named neighbor, specific assignment, specific class topic, specific family member, specific route, specific home feature, or specific prior event is NOT disposable merely because it appears inside casual dialogue.
Named entities and concrete factual claims are unsupported unless they appear in the runtime context or are established by the user's request/current scene.
Do not approve a draft merely because an invented detail is plausible or sounds like normal teen conversation.

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

ALLOWED TOP-LEVEL PATCH FIELDS:
- status
- chapter
- scene
- scene_completed
- location
- time
- current_situation
- character_knowledge
- completed_events
- active_clues
- new_clues
- unresolved_questions
- active_objectives
- continuity_requirements

Never create any other top-level field. In particular, do NOT use fields such as "conversation_topics", "current_activity", "summary", "notes", "recent_events", or any other field not listed above.

CRITICAL COMPACTNESS RULES:
- Do NOT copy unchanged fields from current_state.json into patch.
- Do NOT repeat existing array items.
- Do NOT rewrite existing arrays merely because they already exist.
- If a field has no new information, leave it out of patch.
- If nothing changed, the patch MUST be {}.
- The normal response should be a small JSON object, not a copy of current_state.json.
- Never infer chapter or scene numbers from prose. When a manuscript section filename is supplied, the application provides the authoritative chapter and scene bookkeeping numbers separately.

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
- Every evidence "quote" must be one contiguous excerpt copied verbatim from the completed story section. Never stitch together separate parts of the section.
- Never use ellipses ("..."), brackets, summaries, paraphrases, or text from CURRENT STATE BEFORE THIS SECTION as part of a quote.
- The quote must directly support the claim and should be short enough to copy exactly.
- If there is no direct contiguous quote supporting a claim, do not put the claim in the patch.
- Existing state facts are not changes just because the completed section mentions or implies them again.
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
- Do not use evidence to turn temporary scene behavior into permanent character knowledge or continuity rules.
- If the story section does not explicitly support a proposed change, do not include that change.
- Do not output markdown, explanations, notes, analysis, or code fences.
"""

pending_state = None
STORY_SAVE_LOCK = threading.Lock()


def next_story_section_path(chapter, scene):
    chapter_dir = MANUSCRIPT_DIR / "Chapters" / f"Chapter_{chapter:02d}"
    candidate_scene = scene
    while True:
        filename = f"Chapter_{chapter:02d}_Section_{candidate_scene:02d}.txt"
        path_out = chapter_dir / filename
        if not path_out.exists():
            return candidate_scene, filename, path_out
        candidate_scene += 1


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


def requested_word_range(user_request):
    match = re.search(
        r"Write\s+approximately\s+([\d,]+)\s+to\s+([\d,]+)\s+words",
        str(user_request or ""),
        re.IGNORECASE
    )

    if not match:
        return None, None

    return (
        int(match.group(1).replace(",", "")),
        int(match.group(2).replace(",", ""))
    )


def story_word_count(text):
    return len(
        re.findall(
            r"\b[\w’'-]+\b",
            str(text or "")
        )
    )


def scene_requires_revision(user_request, draft, review):
    """Apply deterministic gates before trusting the local-model reviewer."""
    requested_min, requested_max = requested_word_range(user_request)

    if requested_min is not None and story_word_count(draft) < requested_min:
        actual = story_word_count(draft)
        return f"Draft is too short: {actual} words; minimum requested is {requested_min}."

    if not isinstance(review, dict) or review.get("approved") is not True:
        violations = (
            review.get("violations", [])
            if isinstance(review, dict)
            else []
        )
        if violations:
            return "Reviewer rejected the draft: " + "; ".join(str(v) for v in violations[:3])
        return "Reviewer did not approve the draft."

    draft_lower = str(draft or "").lower()
    opening = re.sub(
        r"\s+",
        " ",
        str(draft or "").strip()
    ).lower()[:500]

    # Never allow a meta response to reach the manuscript.
    meta_markers = [
        "the scene goal is",
        "the scene focuses on",
        "the characters should",
        "the story should",
        "in this scene",
        "this scene will",
        "the scene is about",
    ]

    for marker in meta_markers:
        if marker in opening:
            return f"Meta-response detected near the opening: '{marker}'."

    # Deterministic backstop for newly invented titled people.
    unsupported_people = _find_unsupported_named_people(draft)
    if unsupported_people:
        names = ", ".join(sorted(set(unsupported_people))[:3])
        return f"Unsupported named person introduced: {names}."

    # Parse the explicit hard boundary from the user's request.
    avoid = ""
    match = re.search(
        r"DO NOT ADVANCE YET:\s*(.*?)(?:\n\s*\n[A-Z][A-Z /_-]+:|\nOutput only|$)",
        str(user_request or ""),
        re.IGNORECASE | re.DOTALL
    )

    if match:
        avoid = match.group(1).lower()

    # A scene explicitly held before arriving home must not contain concrete
    # arrival language or doorway/house-entry beats.
    if (
        "have not reached home" in avoid
        or "have not reached home yet" in avoid
    ):
        forbidden_location_terms = [
            "reached home",
            "reached the house",
            "reached their house",
            "arrived home",
            "arrived at the house",
            "front door",
            "screen door",
            "entryway",
            "porch steps",
            "porch light",
            "walkway to the house",
            "inside the house",
            "went inside",
            "entered the house",
        ]

        for term in forbidden_location_terms:
            if term in draft_lower:
                return f"Hard scene boundary violated: home/arrival language found ('{term}')."

    # For the explicit Maya/Tony beat used by the scene builder, require the
    # concrete behavioral markers requested by the user. This is intentionally
    # conservative and only activates when those concepts are present in the
    # request.
    request_lower = str(user_request or "").lower()

    if "maria" not in request_lower and "maya" in request_lower and "tony" in request_lower:
        requires_blush = "blush" in request_lower
        requires_evasive = "evasive" in request_lower
        requires_tony_reference = (
            "like tony" in request_lower
            or "likes tony" in request_lower
            or "feelings for tony" in request_lower
            or "crush on tony" in request_lower
        )

        if requires_tony_reference:
            tony_terms = [
                "tony",
                "your dad",
                "your father",
            ]

            if not any(term in draft_lower for term in tony_terms):
                return "Required Maya-to-Tony hint was not detected: the draft does not clearly reference Tony or an unmistakable relationship reference."

        if requires_blush:
            blush_terms = [
                "blush",
                "blushed",
                "blushing",
                "flushed",
                "cheeks warmed",
                "cheeks turned",
                "face warmed",
            ]

            if not any(term in draft_lower for term in blush_terms):
                return "Required Maya blush beat was not detected."

        if requires_evasive:
            evasive_terms = [
                "evasive",
                "avoided the question",
                "changed the subject",
                "deflected",
                "brushed it off",
                "looked away",
                "quickly changed",
                "stammered",
                "stumbled over her words",
            ]

            if not any(term in draft_lower for term in evasive_terms):
                return "Required Maya evasive-behavior beat was not detected."

    return False


def _review_result_is_approved(result):
    return (
        isinstance(result, dict)
        and result.get("approved") is True
    )


def review_story_draft(user_request, draft):
    if session.model_path is None:
        raise RuntimeError(
            "No model is loaded. Use Load Model first."
        )

    runtime_context = build_runtime_story_context(
        session.files
    )

    previous_section = build_previous_section_context()

    requested_min, requested_max = requested_word_range(user_request)

    prompt = f"""Review the following generated story draft for canon, continuity, and task compliance.

USER REQUEST:
{user_request}

RUNTIME STORY CONTEXT:
{runtime_context}

PREVIOUS SAVED STORY SECTION:
{previous_section if previous_section else "No previous saved section is available."}

GENERATED STORY DRAFT:
{draft}

REQUESTED WORD COUNT:
{requested_min if requested_min is not None else "not specified"} to {requested_max if requested_max is not None else "not specified"} words

ACTUAL WORD COUNT:
{story_word_count(draft)}

Return ONLY the required JSON object.
"""

    # The writer and canon reviewer must never hold two copies of the model
    # in memory at the same time on low-memory machines. Save the writer
    # session state, stop it, run the reviewer, then restore the writer.
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
            "-c", "6144",
            "--reasoning", "off",
            "--temp", "0.02",
            "--top-k", "20",
            "--top-p", "0.80",
            "--repeat-last-n", "256",
            "--repeat-penalty", "1.08",
            "--n-predict", "350",
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
                timeout=300
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

        print("\n--- CANON REVIEW ---", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print("--- END CANON REVIEW ---\n", flush=True)

        return result

    finally:
        session.start(
            saved_system_prompt,
            saved_files,
            saved_model_path
        )

def _find_unsupported_named_people(text):
    """
    Deterministic backstop for newly invented titled people such as
    "Mr. Harrison" or "Mrs. Albright". Character names and other known
    proper names are collected from runtime reference context.
    """
    known = set()

    for name, content in session.files:
        try:
            data = json.loads(content)
        except Exception:
            continue

        if isinstance(data, dict):
            for key in ("name", "title"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    known.add(value.strip().lower())

            relationships = data.get("relationships")
            if isinstance(relationships, dict):
                for key in relationships:
                    known.add(str(key).strip().lower())

    matches = re.findall(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Coach|Professor)\.?\s+([A-Z][A-Za-z]+)\b",
        text or "",
    )

    unsupported = []
    for surname in matches:
        normalized = surname.lower()
        if normalized not in known:
            unsupported.append(surname)

    return sorted(set(unsupported))


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

    def ask(self, text):
        with self.lock:
            if self.model_path is None:
                raise RuntimeError(
                    "No model is loaded. Use Load Model first."
                )

            started = time.time()

            full_prompt = self._build_prompt(
                self.system_prompt or SYSTEM_PROMPT,
                self.files
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
                "--prompt", "USER REQUEST:\n" + text,
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

            self.child = proc
            self.started_at = started

            try:
                output, _ = proc.communicate(
                    timeout=300
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate()

                raise TimeoutError(
                    "Story generation timed out. "
                    "The model did not finish within 5 minutes."
                )
            finally:
                self.child = None
                self.buffer = ""

            raw_output = output

            # llama-cli may echo the supplied prompt even when
            # --no-display-prompt is used. The echo is not manuscript content.
            # Strip it using the structural markers that bound our scene
            # packet/continuity context, then fall back to the request text.
            marker_candidates = [
                "[END PREVIOUS SAVED STORY SECTION]",
                "[END SCENE PACKET]",
            ]

            for marker in marker_candidates:
                if marker in raw_output:
                    raw_output = raw_output.rsplit(marker, 1)[1]
                    break
            else:
                prompt_prefixes = [
                    full_prompt + "\nUSER REQUEST:\n" + text,
                    "USER REQUEST:\n" + text,
                    full_prompt,
                ]

                for prefix in prompt_prefixes:
                    if prefix in raw_output:
                        raw_output = raw_output.rsplit(prefix, 1)[1]
                        break

            answer = self.clean_output(raw_output)

            # Final transport-output backstop. A writer response beginning
            # with the scene-builder instructions is contaminated even if the
            # structural marker cleanup above was bypassed.
            prompt_start_markers = (
                "Begin Chapter ",
                "SCENE GOAL:",
                "CHARACTERS:",
                "REQUIRED:",
                "DO NOT ADVANCE YET:",
            )

            for marker in prompt_start_markers:
                if answer.startswith(marker):
                    request_pos = answer.find("DO NOT ADVANCE YET:")
                    end_pos = answer.find("\n", request_pos)
                    if request_pos >= 0 and end_pos >= 0:
                        answer = answer[end_pos + 1:].lstrip()
                    else:
                        answer = ""
                    break

            if not answer:
                raise RuntimeError(
                    "llama-cli returned an empty or prompt-contaminated response."
                )

            return answer, time.time() - started

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

                started = time.time()

                answer, measured = session.ask(
                    text
                )

                requested_min, requested_max = requested_word_range(text)

                # Short sections are intentional. Do not spend another
                # expensive model pass trying to force an arbitrary word minimum.
                try:
                    review = review_story_draft(
                        text,
                        answer
                    )
                except Exception as exc:
                    print(
                        "Scene review skipped: " + str(exc),
                        flush=True
                    )
                    review = {
                        "approved": True,
                        "violations": []
                    }

                needs_revision = scene_requires_revision(
                    text,
                    answer,
                    review
                )

                if needs_revision:
                    violations = (
                        review.get("violations", [])
                        if isinstance(review, dict)
                        else []
                    )

                    revision_lines = [
                        "WRITE THE SCENE NOW.",
                        "Return only the replacement story prose.",
                        "Do not describe the assignment, summarize the scene, explain what should happen, or output an outline.",
                        "The first line of the response must be narrative prose or dialogue.",
                    ]

                    if requested_min is not None:
                        revision_lines.append(
                            f"Write at least {requested_min} words and stay within the requested range when practical."
                        )

                    if violations:
                        revision_lines.append(
                            "FIX THESE SPECIFIC REVIEW PROBLEMS IN THE REPLACEMENT:"
                        )
                        for violation in violations[:3]:
                            if isinstance(violation, dict):
                                quote = str(
                                    violation.get("quote", "")
                                ).strip()
                                reason = str(
                                    violation.get("reason", "")
                                ).strip()
                                category = str(
                                    violation.get("category", "")
                                ).strip()

                                detail = (
                                    "- Remove or rewrite this unsupported detail"
                                    + (f" [{category}]" if category else "")
                                    + (f': "{quote}"' if quote else "")
                                    + (f" Reason: {reason}" if reason else "")
                                )
                            else:
                                detail = f"- {str(violation).strip()}"

                            if detail.strip() != "-":
                                revision_lines.append(detail)

                        revision_lines.append(
                            "Do not replace one unsupported named person with another. Keep the same scene boundaries and required beats."
                        )

                    revision_request = (
                        text
                        + "\n\n"
                        + "\n".join(revision_lines)
                    )

                    revised_answer, revised_elapsed = session.ask(
                        revision_request
                    )

                    answer = revised_answer
                    measured += revised_elapsed

                    final_failure = scene_requires_revision(
                        text,
                        answer,
                        {"approved": True}
                    )
                    if final_failure:
                        raise ValueError(
                            "The model produced a draft that does not satisfy "
                            "the requested scene boundaries or required beats. "
                            f"Reason: {final_failure} "
                            "The draft was not saved."
                        )

                elapsed = measured

                self._json(200, {
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
