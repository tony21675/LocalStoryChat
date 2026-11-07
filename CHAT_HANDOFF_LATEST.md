# LocalStoryChat handoff, current checkpoint

Date: October 31, 2026

## Current repository status

This repository remains the existing **LocalStoryChat v7** project: a local browser chat UI built around the llama.cpp / `llama-cli --simple-io` workflow. Its current story-oriented behavior and files have not been changed as part of the future-app planning.

Existing launch:
```bash
cd ~/Documents/LocalStoryChat
./start.sh
```

Existing desktop installer:
```bash
cd ~/Documents/LocalStoryChat
./install_desktop.sh
```

## Future App 3: standalone Local Model Chat

The author wants a separate, simple chat window for talking with installed local models, similar to an ordinary ChatGPT-style conversation.

Intended direction:
- Let the author choose among locally installed models.
- Provide a straightforward message / response interface.
- Preserve context within a conversation and allow a fresh conversation to be started.
- Stay focused on general conversation, not story writing.

Important boundary:
- Do not automatically read or write `~/Documents/MyNovel`, `~/StoryFiles`, manuscripts, character cards, Story Bible data, or writing guidance.
- Do not add story prompts, scene tools, validators, or novel-package logic.
- The existing LocalStoryChat v7 code may be reviewed for local model backend lessons, but do not retrofit, replace, or remove its current story-oriented behavior without discussing that separately. The planned general chat app should remain independently scoped.

## Future App 4: DIY Vape Recipe Studio

This is a separate planned local application for the author's DIY e-liquid recipes and flavor inventory. The author has about 15 years of mixing experience, so the tool should be an experienced-user workbench rather than a beginner tutorial.

Personal preferences to preserve:
- Author's profile: flavors such as cotton candy and marshmallow.
- Wife's profile: Super Sweet.
- Keep the two taste profiles distinct.

Possible feature scope:
- Save, duplicate, edit, and version recipes, retaining known-good versions.
- Batch size, flavor percentages and amounts, VG/PG targets, and carrier contribution from flavorings and other liquids.
- Record nicotine product label strength and carrier from user-entered product details; calculations must not guess missing values.
- Flavor stash with brand, flavor name, bottle size, remaining quantity, and personal notes.
- Inventory-aware recipe suggestions, batch usage tracking, and shopping/restock lists.
- Taste ratings, notes, and adjustment history.
- Local-only data with backups and portable JSON/CSV export.
- Investigate importing the author's own E-Liquid Recipes CSV export after confirming the actual export format. The website must not be a dependency.

Keep this recipe application entirely separate from LocalStoryChat, StoryBuilder, and novel data. Its own GitHub repository should be created only when the author is ready to start App 4.

## Priority and hold status

**App 3 and App 4 are both on hold. Do not begin implementation until the author says StoryBuilder Writer and Story Bible are finished.**

Current priority remains:
1. Finish and stabilize StoryBuilder Minimal Writer.
2. Finish and stabilize Story Bible.
3. Complete the agreed desktop launchers for the writing apps.
4. Then resume App 3 and App 4 as separate projects, discussing each design before code changes.

The author eventually wants four easy desktop launchers: Writer, Story Bible, Local Model Chat, and DIY Vape Recipe Studio.

## Related handoff

The shared future-app backlog is maintained only in [StoryBuilder's FUTURE_APPS_ON_HOLD.md](https://github.com/tony21675/StoryBuilder/blob/redesign/simplified-writing-workflow/FUTURE_APPS_ON_HOLD.md). This repository's duplicate copy was removed to prevent divergent versions.

**This handoff records future intent only. It does not authorize starting either app now.**
