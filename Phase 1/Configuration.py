"""Configuration management utilities for the trading research prototype.

This module provides the :class: 'Config' class which centralises how runtime
how runtime options are loaded and accessed across the system. According to the
''README.md'' architecture overview, configuration data is a core dependency
that multiple layers-data acquisition, strategy logic, storage and logging-need
to agree on. Keeping the implementation in one place makes it easier fo the
future subsystems to query shared settings such as API keys, default timeframes
or storage paths.
"""

from __future__ import annotations

import json
import pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

try: # pragma: no cover - optional dependency, kept lightweight
    import yaml
except Exception: # pylint: disable=broad-except
    yaml = None # type: ignore

class Config:
    """Typed wrapper around nested configuration dictionaries.

    The class offers a minimal but robust feature set:

    * :meth: 'load' reads JSON or YAML files from disk and returns a ready-to-use
    :class: 'Config' instance.
    * :meth: 'get' supports dotted keys (''"""