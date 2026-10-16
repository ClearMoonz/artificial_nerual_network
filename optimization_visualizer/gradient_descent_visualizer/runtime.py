"""Configure optional local Qt wheels or a standard installed PyQt6 environment."""
import os
import sys
from pathlib import Path

_handles = []


def configure(offscreen=False):
    project = Path(__file__).resolve().parent.parent
    deps = project / 'software' / ('anaconda_runtime' if sys.version_info[:2] == (3, 13) else 'runtime')
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    import PyQt6
    qt = Path(PyQt6.__file__).resolve().parent / 'Qt6'
    if (qt / 'plugins').is_dir():
        os.environ['QT_PLUGIN_PATH'] = str(qt / 'plugins')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt / 'plugins' / 'platforms')
    os.environ['QT_API'] = 'pyqt6'
    if offscreen:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    if os.name == 'nt' and (qt / 'bin').is_dir():
        _handles.append(os.add_dll_directory(str(qt / 'bin')))
