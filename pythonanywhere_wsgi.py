"""WSGI entry point for a PythonAnywhere web app.

Paste this file into the PythonAnywhere WSGI configuration, or point the
WSGI file setting at this file after cloning the repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(
    os.environ.get("ABP_PROJECT_ROOT", Path.home() / "ABP")
).expanduser()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import app as application  # noqa: E402, F401
