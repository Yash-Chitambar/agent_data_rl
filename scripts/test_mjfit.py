"""Run mjfit's tests without RoboSimStudio's top-level `tests` shadowing them."""

from pathlib import Path
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "mjfit" / "tests"

if __name__ == "__main__":
    # mjfit uses a namespace directory named `tests`; RoboSimStudio's editable
    # install exposes a regular package with that same name. Select mjfit's
    # namespace for this process, including imports between its test modules.
    namespace = ModuleType("tests")
    namespace.__path__ = [str(TESTS)]
    sys.modules["tests"] = namespace
    sys.path.insert(0, str(ROOT / "mjfit"))
    raise SystemExit(pytest.main([str(TESTS), *sys.argv[1:]]))
