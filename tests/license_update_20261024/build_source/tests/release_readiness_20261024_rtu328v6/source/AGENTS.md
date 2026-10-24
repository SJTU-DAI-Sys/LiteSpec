Store debugging, testing, and evaluation scripts and files in the repository's
`tests/` directory.

Before starting a task, explicitly inspect `.agents/skills/*/SKILL.md` in this
repository. The `.agents` directory is hidden; do not rely on default search
results that exclude hidden files. If a task matches a repository skill's
description, read its entire `SKILL.md` and any required references before
proceeding. Follow its confirmation steps and use autonomous fallbacks only
where explicitly allowed.

For downloading, inspecting, extracting prompts from, or converting a new
dataset, use `.agents/skills/prepare-spec-decoding-dataset/SKILL.md`. Before
creating an adapter or running the full conversion, inspect five records.
When interaction is available, confirm both of the following with the user:

1. Which field or path to use for prompt extraction.
2. Whether to use `multi` to retain all user turns or `first` to retain only the
   first user turn.

When interaction is available, wait for explicit answers to both questions
before creating an adapter or running the full conversion. If interaction is
unavailable, choose the highest-confidence non-empty prompt candidate from the
inspected records and default to `multi`, unless the user has already specified
these choices. Report the selected field or path and turn mode, then proceed
with adapter creation, sample validation, and full conversion. Stop if no
credible prompt candidate exists.
