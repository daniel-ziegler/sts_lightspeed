# CLAUDE.md

Repository-level guidance for Claude. See `AGENTS.md` for the general project docs (build,
testing, architecture, etc.); the notes below are Claude-specific rules of engagement.

## No shortcuts

We want FULL CORRECTNESS. No edge cases that don't quite work right. If you can't figure out
how to solve something 100% correctly, please flag it loudly to me. If I've told you to 
proceed autonomously, do so but log the issue in a `CORRECTNESS_ISSUES.md` file and tell me
next time I'm there.

## Comment style

Comments describe the current code, not its history. **Never write "tombstone comments"** that
reference the previous version ("was X, now Y", "the prior two-pass version", "before the
refactor", etc.) — git already records the diff and the explanation rots as soon as the old
version is forgotten.

Useful comments include: a concise summary of what a non-trivial block of code does, a
non-obvious safety invariant, a correctness guarantee, a perf rationale, or a contract the
caller depends on. A one-line summary at the top of a tricky loop or function is fine even
if "obvious" on careful reading — it lets the reader skim. Just keep them present-tense and
self-contained.

When in doubt, ask: "would a reader of *just this code* — without seeing the diff or the prior
session's chat — find this comment useful?" If the comment only makes sense alongside the
removed code, delete it.
