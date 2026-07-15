"""Runtime infrastructure for the AgentAbstain environment system.

Holds the BaseEnvironment contract, the multi-environment composer, and
the auto-generated registry of concrete environments. The sibling
`abstention_factory/environments/` directory is a pure artifact tree:
per-environment Python packages produced by the codegen pipeline, safe
to delete with `rm -rf` between regenerations.

`__init__.py` is intentionally empty. Earlier revisions eagerly
re-exported `multi` and `registry` for ergonomics, but that path
introduces a circular import: every env's `environment.py` does
`from abstention_factory.runtime.base import BaseEnvironment, ToolError`
which initializes `abstention_factory.runtime`, which imports
`runtime.registry`, which eagerly imports every env's `environment`
module — including the very env still being initialized. The first
env in the registry's alphabetical order then fails with
"cannot import name 'XxxEnvironment' from partially initialized
module". Keeping this file empty forces consumers to import from
`abstention_factory.runtime.base` / `runtime.multi` / `runtime.registry`
explicitly, breaking the cycle.
"""
