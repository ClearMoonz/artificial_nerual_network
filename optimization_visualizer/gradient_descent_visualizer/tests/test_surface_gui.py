"""Check 3D rendering against real records and exercise mouse rotation/playback."""
import json
import sys
from pathlib import Path
from time import monotonic
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import configure
configure(True)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtTest import QTest
from matplotlib import font_manager, rcParams
import numpy as np
font='C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(font)
rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
rcParams['axes.unicode_minus']=False
from main_window import MainWindow
from surface_view import DEFAULT_CAMERA

app=QApplication([])
app.setStyle('Fusion')
QFontDatabase.addApplicationFont(font)
win=MainWindow()
win.show()
QTest.qWait(200)
root=Path(__file__).resolve().parents[2]/'results'/'三维验收'


def state():
    return json.dumps([{'history':r.history,'records':r.records,'counts':r.counts,
                        'status':r.status,'cpu':r.cpu_seconds} for r in win.runs],sort_keys=True)


def verify_heights(surface):
    for run in win.runs:
        data=np.array(surface.lines[run.method].get_data_3d()).T
        expected=np.array([[*s['x'],s['f']] for s in run.history])
        np.testing.assert_allclose(data,expected,atol=1e-14)
        np.testing.assert_allclose(data[:,2],win.objective.grid(data[:,0],data[:,1]),atol=1e-14)


try:
    win.step_once()
    original=state()
    identities=[id(r) for r in win.runs]
    win.view_mode.setCurrentIndex(1)
    app.processEvents()
    assert original==state() and identities==[id(r) for r in win.runs]
    assert win.model_axes.name=='3d'
    verify_heights(win.surface)
    win.canvas.draw()
    camera=win.surface.camera
    bbox=win.model_axes.bbox
    ratio=win.canvas.devicePixelRatioF()
    center=QPoint(round((bbox.x0+bbox.width*.55)/ratio),
                  round(win.canvas.height()-(bbox.y0+bbox.height*.5)/ratio))
    QTest.mousePress(win.canvas,Qt.MouseButton.LeftButton,pos=center)
    QTest.mouseMove(win.canvas,center+QPoint(45,25),delay=30)
    QTest.mouseRelease(win.canvas,Qt.MouseButton.LeftButton,pos=center+QPoint(45,25))
    app.processEvents()
    rotated=win.surface.camera
    assert not np.allclose(camera,rotated), 'Dragging failed to rotate the 3D axes'
    assert original==state(), 'Rotation modified algorithm states'
    win.step_once()
    assert np.allclose(win.surface.camera,rotated), 'Animation reset the camera'
    verify_heights(win.surface)
    original=state()
    win.view_mode.setCurrentIndex(0)
    assert win.model_axes.name=='rectilinear' and state()==original
    win.view_mode.setCurrentIndex(1)
    assert state()==original and np.allclose(win.surface.camera,rotated)
    win.open_large_surface()
    app.processEvents()
    assert win.large_surface.isVisible()
    verify_heights(win.large_surface.surface)
    win.step_once()
    verify_heights(win.surface)
    verify_heights(win.large_surface.surface)
    win.delay.setValue(30)
    lengths=[len(r.records) for r in win.runs]
    win.play()
    deadline=monotonic()+8
    while len(win.runs[0].records)==lengths[0] and monotonic()<deadline:
        QTest.qWait(50)
    assert len(win.runs[0].records)>lengths[0], '3D playback did not advance'
    win.pause()
    paused=state()
    QTest.qWait(150)
    assert state()==paused
    # Switching view while running must preserve both the timer and the Run objects.
    win.play()
    win.view_mode.setCurrentIndex(0)
    assert win.timer.isActive() and identities==[id(r) for r in win.runs]
    win.view_mode.setCurrentIndex(1)
    win.pause()
    win.finish()
    deadline=monotonic()+15
    while not all(r.done for r in win.runs) and monotonic()<deadline:
        QTest.qWait(50)
    assert all(r.done for r in win.runs)
    assert [len(r.records) for r in win.runs]==[19,1,2]
    verify_heights(win.surface)
    verify_heights(win.large_surface.surface)
    win.reset_surface_camera()
    assert np.allclose(win.surface.camera,DEFAULT_CAMERA)
    assert np.allclose(win.large_surface.surface.camera,DEFAULT_CAMERA)
    win.export_to(root/'三算法三维')
    for name in ['iterations.csv','experiment.json','plots.png','interface.png','surface3d.png','view.json']:
        assert (root/'三算法三维'/name).stat().st_size>0
    win.large_surface.grab().save(str(root/'三算法三维'/'放大窗口.png'))
    # Reset from the large 3D window must rewind the actual experiment, not just its camera.
    camera = win.large_surface.surface.camera
    QTest.mouseClick(win.large_surface.buttons['reset'],Qt.MouseButton.LeftButton)
    app.processEvents()
    assert not win.timer.isActive()
    assert all(len(r.records)==0 and len(r.history)==1 for r in win.runs)
    for r in win.runs:
        np.testing.assert_allclose(r.x,[.8,-.25],atol=0)
    verify_heights(win.surface)
    verify_heights(win.large_surface.surface)
    assert np.allclose(camera,win.large_surface.surface.camera)
    win.step_once()
    win.play()
    QTest.mouseClick(win.large_surface.buttons['reset'],Qt.MouseButton.LeftButton)
    assert not win.timer.isActive() and all(len(r.records)==0 for r in win.runs)
    win.step_once()
    assert [len(r.records) for r in win.runs]==[1,1,1]
    win.function.setCurrentIndex(2)
    win.presets.setCurrentIndex(3)
    assert win.is_3d and win.surface.clipped
    before=state()
    old_range=win.surface.z_range
    win.focus_surface.setChecked(False)
    assert not win.surface.clipped and win.surface.z_range[1]>old_range[1]
    assert state()==before, 'Valley focus changed the algorithm'
    win.focus_surface.setChecked(True)
    for r in win.runs:
        r.finish()
    win.render()
    app.processEvents()
    assert '鞍点' in win.runs[0].status
    verify_heights(win.surface)
    verify_heights(win.large_surface.surface)
    win.export_to(root/'牛顿法鞍点三维')
    win.large_surface.grab().save(str(root/'牛顿法鞍点三维'/'放大窗口.png'))
    win.large_surface.close()
    win.open_large_surface()
    assert win.large_surface.isVisible()
    print('PASS: true surface heights, 2D/3D state preservation, rotation, playback/pause, reset to starting point, enlarged synchronization, exports, saddle and valley focus')
finally:
    win.close()
