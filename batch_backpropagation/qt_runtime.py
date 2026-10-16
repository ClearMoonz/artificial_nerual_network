"""Load project-local Qt and override wheel paths for non-ASCII directories."""
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path


def configure():
    root = Path(__file__).resolve().parent
    conda_bin = Path(sys.prefix) / 'Library' / 'bin'
    if conda_bin.is_dir():
        os.environ['PATH'] = str(conda_bin) + os.pathsep + os.environ.get('PATH', '')
    deps = root / '.deps'
    # Only use the local wheels when their sip extension supports this Python ABI.
    # Anaconda Python 3.13 must use its own packages instead of the cp312 wheels.
    compatible_sip = PathFinder.find_spec('PyQt5.sip', [str(deps / 'PyQt5')])
    if compatible_sip is not None:
        sys.path.insert(0, str(deps))
        qt = deps / 'PyQt5' / 'Qt5'
        # PyQt5 wheel embeds a lossy path in qt.conf on Chinese Windows paths.
        os.environ['QT_PLUGIN_PATH'] = str(qt / 'plugins')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt / 'plugins' / 'platforms')
        os.environ['QML2_IMPORT_PATH'] = str(qt / 'qml')
