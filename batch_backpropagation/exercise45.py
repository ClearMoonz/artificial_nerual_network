"""Dedicated entry in the original Python / Qt-QML framework."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from qt_runtime import configure
configure()
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QTimer
from nn_sandbox.bridges.batch_bp_bridge import BatchBPBridge


def main():
    os.environ.setdefault('QT_QUICK_CONTROLS_STYLE', 'Default')
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()
    bridge = BatchBPBridge()
    engine.rootContext().setContextProperty('batchBP', bridge)
    engine.load(QUrl.fromLocalFile(str(ROOT / 'nn_sandbox/frontend/exercise45.qml')))
    if not engine.rootObjects():
        return 1
    app.aboutToQuit.connect(bridge.shutdown)
    if '--smoke-test' in sys.argv:
        QTimer.singleShot(1000, app.quit)
    result = app.exec_()
    # Destroy QML bindings while the Python context object is still alive.
    from PyQt5 import sip
    sip.delete(engine)
    return result


if __name__ == '__main__':
    sys.exit(main())
