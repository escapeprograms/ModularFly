# Code Design
Work arrives in small steps, and later tasks will build on what you write now.

## Planning
For tasks that add or change structure (not small fixes), include a "Design" section before the steps:
- Each new module/class and its single job
- The interfaces between them (function signatures or data shapes)
- Which existing abstractions you're reusing instead of creating new ones
- Where the next likely feature would plug in, and what would need to change

If the existing structure doesn't fit this task cleanly, say so. Either propose the smallest refactor that makes it fit, as a separate step before the feature (tests passing after it), or take a shortcut and record it as debt. Don't refactor beyond what this task needs.

Check `debt.md` for entries whose "Fix when" matches this task; if one does, propose fixing it as a separate first step.

## Debt
- Single-spot debt: `# TODO(debt): <what to fix>` at that spot.
- Debt spanning several files: an entry in `debt.md` (follow the template there) plus `# TODO(debt-NNN)` at each spot.
- If you notice unrelated debt in code you're already changing, don't fix it; record it and mention it in your summary.

## While coding
- Add a new abstraction (base class, registry, factory) only when there are two real uses, or the plan names the next one.
- Add new behavior as a new function or class, not as `if mode == ...` branches or boolean flags inside existing functions.
- Keep functions short (about 40 lines max) with one job each; split rather than nest.
- Don't copy-paste more than a few lines; pull the shared part into a helper.
- Keep file, network and command-line code separate from core logic.
- Put values a later task may change (paths, sizes, model names) in config, not inline.
- Don't add try/except or fallback code that hides errors.
- If you change a public interface, update every caller and list them in your summary.

# Conda Environment
modular-fly

# Documentation
In the folders that you edited, update the README inside that details what each file and function does. Be concise, but informative in your descriptions, so that any person reading it can understand what it does.

# Agents
The orchestrator should be Fable. Agents for exploring code and coding should be Opus.