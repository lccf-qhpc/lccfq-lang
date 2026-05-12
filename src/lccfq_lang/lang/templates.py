"""Named sub-circuit registry.

Users register named sub-circuits with `register(name, builder)`, where `builder`
has signature (isa, target, **kwargs) -> List[Instruction]. The dispatched
`template` function looks up a builder by name (passed as the `name` kwarg
through BlockFactory.block) and forwards remaining kwargs to it.
"""

from typing import Callable, Dict, List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


Builder = Callable[..., List[Instruction]]

_REGISTRY: Dict[str, Builder] = {}


def register(name: str, builder: Builder) -> None:
    """Register a named sub-circuit builder.

    :raises ValueError: if `name` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Template '{name}' is already registered")
    _REGISTRY[name] = builder


def unregister(name: str) -> None:
    """Remove a named sub-circuit builder.

    :raises KeyError: if `name` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Template '{name}' is not registered")
    del _REGISTRY[name]


def get(name: str) -> Builder:
    """Return the builder registered under `name`.

    :raises KeyError: if `name` is not registered, with a message listing
        currently-available names.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Template '{name}' is not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def template(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Dispatched form: pop `name` from kwargs and invoke the registered builder."""
    name = kwargs.pop("name")
    builder = get(name)
    return builder(isa, target, **kwargs)
