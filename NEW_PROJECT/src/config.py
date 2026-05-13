from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value == "{}":
        return {}
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    raw_lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, raw.strip()))

    def parse_block(start: int, indent: int) -> tuple[Any, int]:
        if start >= len(raw_lines):
            return {}, start
        is_list = raw_lines[start][1].startswith("- ")
        if is_list:
            values = []
            i = start
            while i < len(raw_lines):
                line_indent, content = raw_lines[i]
                if line_indent != indent or not content.startswith("- "):
                    break
                item = content[2:].strip()
                if item:
                    values.append(_parse_scalar(item))
                    i += 1
                else:
                    child, i = parse_block(i + 1, raw_lines[i + 1][0])
                    values.append(child)
            return values, i

        mapping: Dict[str, Any] = {}
        i = start
        while i < len(raw_lines):
            line_indent, content = raw_lines[i]
            if line_indent != indent or content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"Invalid config line: {content}")
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                mapping[key] = _parse_scalar(value)
                i += 1
            else:
                if i + 1 >= len(raw_lines) or raw_lines[i + 1][0] <= line_indent:
                    mapping[key] = None
                    i += 1
                else:
                    child, i = parse_block(i + 1, raw_lines[i + 1][0])
                    mapping[key] = child
        return mapping, i

    parsed, _ = parse_block(0, raw_lines[0][0] if raw_lines else 0)
    return parsed or {}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = _simple_yaml_load(text)

    project_root = path.parent.parent
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root)
    return cfg


def resolve_project_path(cfg: Dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(cfg["_project_root"]).resolve() / path


def merged_training_params(cfg: Dict[str, Any], model_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params.update(deepcopy(cfg.get("training", {})))
    params.update(deepcopy(cfg.get("model_params", {}).get(model_name, {})))
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params
