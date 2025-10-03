"""Configuration management utilities for the trading research prototype.

This module provides the :class: 'Config' class which centralises how
runtime options are loaded and accessed across the system. According to the
''README.md'' architecture overview, configuration data is a core dependency
that multiple layers-data acquisition, strategy logic, storage and logging-need
to agree on. Keeping the implementation in one place makes it easier fo the
future subsystems to query shared settings such as API keys, default timeframes
or storage paths.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

try: # pragma: no cover - optional dependency, kept lightweight
    import yaml
except Exception: # pylint: disable=broad-except
    yaml = None # type: ignore

class Config:
    """Typed wrapper around nested configuration dictionaries.

    The class offers a minimal but robust feature set:

    * :methode: 'load' reads JSON or YAML files from disk and returns a ready-to-use
    :class: 'Config' instance.
    * :methode: 'get' supports dotted keys ('' "database.host" '') to access nested
    structures with an optional default value when a key is missing.
    * :methode: 'as_dict' exposes a deep copy of the underlying mapping when raw
    access is required by downstream components.

    Examples
    --------
    >>> cfg = Config.load("settings.json")
    >>> cfg.get("exchange.api_key")
    'abc1232'
    >>> cfg.get("nonexistent", default = "fallback")
    'fallback'
    """

    _SUPPORTED_SUFFIXES: Mapping[str, str] = {".json": "json", ".yml": "yaml", ".yaml": "yaml"}

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = dict(data or {})

    # --------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: "Path | str") -> "Config":
        """Load configuration data from *path*.

        Parameters
        ----------
        path:
            File path to a JSON or YAML configuration file.

        Returns
        -------
        Config
            A new :class: 'Config' instance populated with the file's content.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist.
        ValueError
            When the file suffix is not supported or YAML parsing is requested
            without ''PyYAML'' being installed.
        json.JSONDecodeError
            In case the input file contains invalid JSON.
        yaml.YAMLError
            In case the input file contains invalid YAML (when available).
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()
        format_name = cls._SUPPORTED_SUFFIXES.get(suffix)
        if format_name is None:
            supported = ", ".join(sorted(cls._SUPPORTED_SUFFIXES))
            raise ValueError(
                f"Unsupported configuration file format '{suffix}'. Expected one of: {supported}"
            )

        if format_name == "json":
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else: #YAML
            if yaml is None: # pragma: no cover - environment dependent
                raise ValueError(
                    "YAML configuration requested but PyYAML is not installed. Install PyYAML to use YAML configs."
                )
            with path.open("r", encoding="utf-8") as fh: # pragma: no cover - requires optional dependency
                data = yaml.safe_load(fh) or {}

        if not isinstance(data, MutableMapping):
            raise ValueError("Configuration root must be a mapping/dictionary.")

        return cls(data)

    # -------------------------------------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return the configuration value for *key*.

        The method accepts dotted keys to traverse nested dictionaries.
        When a key cannot be resolved the *default* is returned instead of raising an error.
        This behaviour allows optional settings throughout the system to read configuration values safely.
        """

        if not key:
            return default

        current: Any = self._data
        for fragment in key.split("."):
            if isinstance(current, Mapping) and fragment in current:
                current = current[fragment]
            else:
                return default
        return current

    # -------------------------------------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """ Return the copy of the underlying dictionary.

        Mutating the returned mapping does not affect the :class: 'Config' instance,
        keeping configuration reads side effect free.
        """

        return json.loads(json.dumps(self._data))

    # ------------------------------------------------------------------------------------------------
    def keys(self) -> Iterable[str]:
        """Iterate over top-level keys for convenience."""

        return self._data.keys()

    # ------------------------------------------------------------------------------------------------
    def __contains__(self, key: object) -> bool: # pragma: no cover - trivial
        return key in self._data

    # ------------------------------------------------------------------------------------------------
    def __repr__(self) -> str: # pragma: no cover - trivial
        return f"Config({self._data!r})"

# ----------------------------------------------------------------------------------------------------
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")

def _resolve_env_placeholders(value: Any) -> Any:
    """Recursively resolve ${VAR} placeholders using environment variable."""

    if isinstance(value, Mapping):
        return {key: _resolve_env_placeholders(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default
            raise KeyError (
                f"Environment variable '{var_name}' required by configuration but not set."
            )
        return _ENV_PATTERN.sub(_replace, value)
    return value

@dataclass(frozen=True)
class BinanceSettings:
    api_key: str
    api_secret: str
    base_url: str

@dataclass(frozen=True)
class MySqlSettings:
    host: str
    port: int
    user: str
    password: str
    database: str

@dataclass(frozen=True)
class Settings:
    """Container bundling parsed configuration sections."""

    config: Config
    binance: BinanceSettings
    mysql: MySqlSettings

def load_settings(path: str) -> Settings:
    """"Load configuration from *path* and return typed settings sections."""

    config = Config.load(path)
    resolved_data = _resolve_env_placeholders(config.as_dict())
    resolved_config = Config(resolved_data)

    binance_raw = resolved_config.get("exchange.binance")
    if not isinstance(binance_raw, Mapping):
        raise ValueError("Configuration is missing the 'exchange.binance' section.")

    mysql_raw = resolved_config.get("database.mysql")
    if not isinstance(mysql_raw, Mapping):
        raise ValueError("Configuration is missing the 'database.mysql' section.")

    try:
        binance = BinanceSettings(
            api_key=str(binance_raw["api_key"]),
            api_secret=str(binance_raw["api_secret"]),
            base_url=str(binance_raw["base_url"]),
        )
    except KeyError as exc: # pragma: no cover - defensive
        raise ValueError("Incomplete Binance configuration.") from exc

    try:
        mysql = MySqlSettings(
            host=str(mysql_raw["host"]),
            port=int(mysql_raw["port"]),
            user=str(mysql_raw["user"]),
            password=str(mysql_raw["password"]),
            database=str(mysql_raw["database"]),
        )
    except KeyError as exc: # pragma: no cover -defensive
        raise ValueError("Incomplete MySQL configuration.") from exc

    return Settings(config=resolved_config, binance=binance, mysql=mysql)
