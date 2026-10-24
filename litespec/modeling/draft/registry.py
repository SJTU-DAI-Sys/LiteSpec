"""Draft-architecture registry: the extension point for new draft model classes.

A draft *architecture* (the model class) is a separate axis from an immutable
algorithm registration: one architecture can train under several algorithms
and vice versa. Registering a class here makes it constructible from a
draft-config JSON whose
``architectures[0]`` names it — adding an architecture is a new file plus
``@register_draft``, not an edit to ``modeling/auto.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DRAFT_REGISTRY: Dict[str, type] = {}


def method_config(config) -> dict:
    """Resolve exactly the configuration owned by the declared architecture."""
    (architecture,) = config.architectures
    cls = resolve_draft(architecture)
    key = cls.method_config_key
    value = getattr(config, key, None)
    if not isinstance(value, dict):
        raise ValueError(f"{architecture} requires {key}")
    other_keys = {
        "dflash_config",
        "dflash2_config",
        "dspark_config",
        "dlite_config",
    } - {key}
    if any(hasattr(config, other) for other in other_keys):
        raise ValueError(f"{architecture} only accepts {key}")
    return value


def draft_source_files(architecture: str) -> tuple[str, ...]:
    """Return the complete, flat remote-code dependency closure."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent
    pending = [resolve_draft(architecture).__module__.rsplit(".", 1)[-1] + ".py"]
    seen = set()
    while pending:
        filename = pending.pop()
        if filename in seen:
            continue
        source = root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing draft export dependency: {source}")
        seen.add(filename)
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or "." in node.module:
                    raise ValueError(
                        f"Draft exports require flat relative imports: {filename}"
                    )
                pending.append(node.module + ".py")
    return tuple(sorted(seen))


def register_draft(cls: Optional[type] = None, *, name: Optional[str] = None):
    """Class decorator registering a draft architecture.

    The key defaults to the class name — exactly what draft-config JSONs carry
    in ``architectures``. The class must declare ``config_class`` (the HF
    ``PretrainedConfig`` subclass it is built from) so the auto loaders can
    resolve both the model and its config from the one registry entry.
    """

    def _register(cls: type) -> type:
        key = name or cls.__name__
        if getattr(cls, "config_class", None) is None:
            raise TypeError(
                f"@register_draft: {cls.__name__} must declare config_class"
            )
        existing = DRAFT_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"draft architecture {key!r} already registered to {existing.__name__}"
            )
        DRAFT_REGISTRY[key] = cls
        return cls

    return _register(cls) if cls is not None else _register


def resolve_draft(name: str) -> type:
    try:
        return DRAFT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown draft architecture {name!r}; available: {available_drafts()}"
        ) from None


def available_drafts() -> List[str]:
    return sorted(DRAFT_REGISTRY)


__all__ = ["DRAFT_REGISTRY", "register_draft", "resolve_draft", "available_drafts"]
