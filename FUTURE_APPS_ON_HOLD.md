# Future standalone applications: on hold

Last updated: October 31, 2026

## Status and priority

**Both future apps are intentionally on hold. Do not begin implementation until the author says StoryBuilder Writer and Story Bible are finished.**

Current priority:
1. Finish and stabilize StoryBuilder Minimal Writer.
2. Finish and stabilize the Story Bible companion.
3. Complete the agreed desktop launchers for the finished apps.
4. Only then resume the two independent applications below, one at a time.

These are not features to fold into StoryBuilder. Each should eventually have its own project/repository, data, settings, and desktop launcher. Keep their scope separate from novel packages.

## App 3: Local Model Chat

A simple desktop chat window for talking to a locally installed language model, similar to an ordinary ChatGPT-style conversation.

### Intended scope
- Choose from the local models already installed on the computer.
- Send messages and receive responses in a straightforward chat interface.
- Maintain conversation context while a chat is in progress.
- Start a fresh conversation when desired.
- Keep the interface focused on conversation, not writing-project management.

### Separation requirements
- No Story Bible, novel package, manuscript, scene, character-card, validator, or writing-guidance integration.
- Do not automatically read or write files inside `~/Documents/MyNovel`.
- Keep it local-first and able to converse without a cloud account or remote API.
- The existing `LocalStoryChat` v7 repository is a useful reference for its local model / llama.cpp experience, but it is an older story-oriented app. Do not retrofit, replace, or remove its current behavior without discussing that separately first. The new generic chat app should have a clean, independent scope.

## App 4: DIY Vape Recipe Studio

A local-first recipe and inventory manager for the author's DIY e-liquid mixing. The author has about 15 years of mixing experience; this should be an experienced-user workbench, not a beginner's mixing tutorial.

### Personal taste profiles
- Author's profile: favors flavors such as cotton candy and marshmallow.
- Wife's profile: favors Super Sweet.
- Keep each person's taste preferences distinct and use the selected profile when organizing or suggesting recipes.

### Recipe and formulation workspace
- Create, save, duplicate, edit, and compare recipes without overwriting a known-good version.
- Keep recipe versions, notes, taste ratings, and planned adjustments.
- Set batch size and record flavor percentages and ingredient amounts.
- Track target and resulting VG/PG balance.
- Account for the VG/PG carrier contributed by each flavoring and other liquid ingredient.
- Record nicotine product details, including user-entered labeled strength and carrier, and account for those entered values in the formulation.
- Support the author's preferred units and, if desired, mixing by weight using recorded ingredient densities.
- Show recipe totals and highlight incomplete or inconsistent inputs instead of silently guessing.

### Flavor inventory
- Track flavor name, brand, bottle size, amount remaining, and personal notes.
- Record which flavors are available and which need replenishing.
- Optionally update inventory when the author records a batch as mixed.
- Generate a shopping / restock list.
- Suggest combinations using flavors actually in stock and the selected person's taste profile.

### Data ownership and E-Liquid Recipes reference
- Store recipes and inventory locally on the author's computer. No account or online service should be required for normal use.
- Include local backups and portable JSON/CSV export so the collection is not trapped in the application.
- Investigate importing the author's own E-Liquid Recipes exports (such as CSV) when development begins; confirm the actual export format before promising compatibility.
- E-Liquid Recipes is a feature reference, not a runtime dependency. The local app must remain useful if that website disappears.
- Calculations should use explicit user-entered ingredient labels, concentrations, carriers, units, and densities. Never invent missing product specifications or silently assume a concentration.

### Project boundary
- Create a dedicated repository for this app when it is ready to begin. No such dedicated repository has been created yet.
- Keep the vape recipe data and code separate from StoryBuilder, LocalStoryChat, novel repositories, and writing data.

## Desktop launchers

When the four apps are eventually ready, the author wants desktop launchers so each can be opened by double-clicking, without remembering terminal commands or project paths.

Planned launchers:
1. StoryBuilder Writer
2. Story Bible
3. Local Model Chat
4. DIY Vape Recipe Studio

## Resume checklist

Before starting App 3 or App 4:
- Confirm Writer and Story Bible are considered finished by the author.
- Review the latest handoff in the relevant repository.
- Discuss the next app's design before making code changes.
- Keep each new app independent and build/test incrementally.
- Add its repository and launcher only when the project is actually started.

**No implementation of App 3 or App 4 is authorized by this handoff. This is a parking note, not a build request.**
