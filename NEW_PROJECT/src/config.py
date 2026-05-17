from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

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


def _load_yaml_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _simple_yaml_load(text)


def _resolve_config_reference(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    root = Path(project_root).resolve()
    project_relative = root / path
    if project_relative.exists():
        return project_relative
    return root / "configs" / path


def default_config_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]

    env_value = os.environ.get("NEW_PROJECT_CONFIG")
    if env_value:
        return _resolve_config_reference(env_value, root)

    active_path = root / "configs" / "active_config.yaml"
    if active_path.exists():
        active_cfg = _load_yaml_file(active_path)
        active_value = active_cfg.get("active_config") or active_cfg.get("config")
        if active_value:
            return _resolve_config_reference(str(active_value), root)

    return root / "configs" / "data2025.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path).resolve() if config_path is not None else default_config_path()
    cfg = _load_yaml_file(path)

    project_root = path.parent.parent
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(project_root)
    return cfg


def resolve_project_path(cfg: Dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(cfg["_project_root"]).resolve() / path


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    else:
        try:
            candidates = list(value)
        except TypeError:
            candidates = [value]

    seen = set()
    out = []
    for item in candidates:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def column_combinations(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    combos: Dict[str, Dict[str, Any]] = {}

    raw = cfg.get("column_combinations") or {}
    if isinstance(raw, list):
        iterable = enumerate(raw, start=1)
    else:
        iterable = raw.items()

    for raw_id, raw_cfg in iterable:
        if not isinstance(raw_cfg, dict):
            continue
        combo_id = str(raw_id)
        name = str(raw_cfg.get("name") or combo_id)
        center = raw_cfg.get("center")
        if not center:
            continue
        combo = {
            "id": combo_id,
            "name": name,
            "center": str(center),
            "exclude_columns": _normalize_string_list(raw_cfg.get("exclude_columns") or raw_cfg.get("exclude")),
        }
        combos[combo_id] = combo
        combos[name] = combo

    return combos


def resolve_center_spec(cfg: Dict[str, Any], value: str) -> Dict[str, Any]:
    text = str(value).strip()
    combos = column_combinations(cfg)
    if text in combos:
        combo = combos[text]
        return {
            "kind": "combo",
            "id": combo["id"],
            "name": combo["name"],
            "label": f"combo{combo['id']}_{combo['name']}",
            "center": combo["center"],
            "exclude_columns": list(combo["exclude_columns"]),
        }
    return {
        "kind": "center",
        "id": "",
        "name": text,
        "label": text,
        "center": text,
        "exclude_columns": [],
    }


def merged_training_params(cfg: Dict[str, Any], model_name: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params.update(deepcopy(cfg.get("training", {})))
    params.update(deepcopy(cfg.get("model_params", {}).get(model_name, {})))
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params
