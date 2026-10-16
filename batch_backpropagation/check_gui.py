"""Offscreen integration check of QML loading and threaded training."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
from qt_runtime import configure
configure()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_QUICK_BACKEND'] = 'software'
os.environ['QT_QPA_FONTDIR'] = str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts')
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtCore import QUrl, QTimer, QMetaObject, QObject
from PyQt5.QtQuick import QQuickWindow
from nn_sandbox.bridges.batch_bp_bridge import BatchBPBridge

app = QApplication([])
app.setQuitOnLastWindowClosed(False)
engine = QQmlApplicationEngine()
bridge = BatchBPBridge()
engine.rootContext().setContextProperty('batchBP', bridge)
engine.load(QUrl.fromLocalFile(str(ROOT / 'nn_sandbox/frontend/exercise45.qml')))
if not engine.rootObjects():
    raise SystemExit('QML load failed')
success = False
grab_result = None
sizes = [(1100, 850), (1600, 1000), (1200, 1100), (900, 650)]
capture_index = 0

def capture():
    global grab_result
    assert not bridge.busy
    assert '达到误差阈值' in bridge.message, bridge.message
    grab_result = engine.rootObjects()[0].contentItem().grabToImage()
    assert grab_result is not None
    grab_result.ready.connect(save_image)

def save_image():
    global success, capture_index
    image = grab_result.image()
    assert not image.isNull(), 'Empty QML screenshot'
    width, height = sizes[capture_index]
    filename = 'interface.png' if capture_index == 0 else f'interface-{width}x{height}.png'
    assert image.save(str(ROOT / 'results' / filename))
    window = engine.rootObjects()[0]
    chart = window.findChild(QObject, 'chartPanel')
    panel = window.findChild(QObject, 'dataPanel')
    assert chart.property('width') > 300 and chart.property('height') >= 300
    assert panel.property('width') >= 300
    print(f'GUI resize OK: {width}x{height}, chart {chart.property("width")}x{chart.property("height")}, data width {panel.property("width")}')
    capture_index += 1
    if capture_index < len(sizes):
        window.setWidth(sizes[capture_index][0])
        window.setHeight(sizes[capture_index][1])
        QTimer.singleShot(300, capture)
    else:
        print('GUI integration OK: QML, worker, results, four window sizes')
        success = True
        app.quit()

def done():
    if not bridge.busy and bridge.worker:
        QTimer.singleShot(300, capture)

bridge.changed.connect(done)
QTimer.singleShot(100, lambda: QMetaObject.invokeMethod(
    engine.rootObjects()[0].findChild(QObject, 'startButton'), 'clicked'))
QTimer.singleShot(20000, app.quit)
app.aboutToQuit.connect(bridge.shutdown)
print('GUI check: entering event loop', flush=True)
app.exec_()
print('GUI check finished; success=', success, 'message=', bridge.message, flush=True)
raise SystemExit(0 if success else 1)
