from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


ToolHandler = Callable[..., Any]


@dataclass
class ToolRegistry:
    """Binds model-visible tool schemas to executable handlers."""

    _definitions: list[dict[str, Any]] = field(default_factory=list)
    _handlers: dict[str, ToolHandler] = field(default_factory=dict)

    def register(self, definition: dict[str, Any], handler: ToolHandler) -> None:
        try:
            name = str(definition["function"]["name"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Tool definition must contain function.name") from exc
        if not name:
            raise ValueError("Tool name cannot be empty")
        if name in self._handlers:
            raise ValueError(f"Tool is already registered: {name}")
        self._definitions.append(deepcopy(definition))
        self._handlers[name] = handler

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return deepcopy(self._definitions)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            output = handler(**arguments)
            return {"ok": True, "output": output}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @classmethod
    def from_definitions(
        cls,
        definitions: Iterable[dict[str, Any]],
        handlers: dict[str, ToolHandler],
    ) -> "ToolRegistry":
        registry = cls()
        definition_names: set[str] = set()
        for definition in definitions:
            try:
                name = str(definition["function"]["name"])
            except (KeyError, TypeError) as exc:
                raise ValueError("Tool definition must contain function.name") from exc
            definition_names.add(name)
            if name not in handlers:
                raise ValueError(f"No handler registered for tool definition: {name}")
            registry.register(definition, handlers[name])
        unused = set(handlers) - definition_names
        if unused:
            raise ValueError(f"Handlers lack tool definitions: {sorted(unused)}")
        return registry
