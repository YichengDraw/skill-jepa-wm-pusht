from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_IMPL = _ROOT.parent / "src" / "skill_jepa"

__path__ = [str(_ROOT), str(_SRC_IMPL)]

