"""Exercise real widget signals, Qt timers, plot clicks and exports offscreen."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import configure
configure(True)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtTest import QTest
from matplotlib import font_manager, rcParams
font='C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(font)
rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
rcParams['axes.unicode_minus']=False
import numpy as np
from main_window import MainWindow
from objectives import OBJECTIVES
from optimizers import Run, Settings

app=QApplication([])
QFontDatabase.addApplicationFont(font)
app.setStyle('Fusion')
win=MainWindow()
win.show()
QTest.qWait(300)

def click(key):
    QTest.mouseClick(win.buttons[key],Qt.MouseButton.LeftButton)
    app.processEvents()


def wait_until_finished():
    from time import monotonic
    deadline=monotonic()+12
    while not all(r.done for r in win.runs) and monotonic()<deadline:
        QTest.qWait(50)
    assert all(r.done for r in win.runs), 'Completion timer timed out'


try:
    assert len(win.runs)==3 and all(len(r.records)==0 for r in win.runs)
    click('step')
    assert [len(r.records) for r in win.runs]==[1,1,1]
    assert win.runs[1].done
    expected=Run(OBJECTIVES['rotated'],'sd',[.8,-.25])
    expected.step()
    np.testing.assert_allclose(win.runs[0].x,expected.x,atol=1e-14)
    win.delay.setValue(30)
    click('run')
    QTest.qWait(250)
    click('pause')
    assert not win.timer.isActive()
    lengths=[len(r.records) for r in win.runs]
    assert lengths[0]>1
    QTest.qWait(100)
    assert lengths==[len(r.records) for r in win.runs], 'Paused states changed'
    click('finish')
    wait_until_finished()
    assert all(r.done for r in win.runs)
    assert [len(r.records) for r in win.runs]==[19,1,2]
    expected.finish()
    np.testing.assert_allclose(win.runs[0].x,expected.x,atol=1e-15)
    root=Path(__file__).resolve().parents[2]/'results'/'界面验收'
    win.export_to(root/'三算法对比')
    for filename in ['iterations.csv','experiment.json','plots.png','interface.png']:
        assert (root/'三算法对比'/filename).stat().st_size>0
    click('reset')
    assert all(len(r.records)==0 for r in win.runs)
    # Click the actual Qt canvas, with axes coordinates converted to widget pixels.
    win.canvas.draw()
    px,py=win.path_axes.transData.transform((-.6,.4))
    point=QPoint(round(px/win.canvas.devicePixelRatioF()),
                 round(win.canvas.height()-py/win.canvas.devicePixelRatioF()))
    QTest.mouseClick(win.canvas,Qt.MouseButton.LeftButton,pos=point)
    app.processEvents()
    assert abs(win.start1.value()+.6)<.02 and abs(win.start2.value()-.4)<.02
    assert all(len(r.records)==0 for r in win.runs)
    # Parameter changes must stop running and discard old trajectories.
    click('run');QTest.qWait(100)
    win.lr.setValue(.02)
    assert not win.timer.isActive() and all(len(r.records)==0 for r in win.runs)
    win.function.setCurrentIndex(2)
    assert len(win.runs)==1 and win.runs[0].method=='newton'
    assert not win.algorithm.model().item(3).isEnabled()
    win.presets.setCurrentIndex(3)  # Figure 9.8 saddle start.
    click('finish');wait_until_finished()
    assert win.runs[0].done and '鞍点' in win.runs[0].status
    win.export_to(root/'牛顿法鞍点')
    win.function.setCurrentIndex(0)
    win.presets.setCurrentIndex(5)  # alpha .041, divergence demonstration.
    assert win.lr.value()==.041 and win.mode.currentData()=='fixed'
    click('finish');wait_until_finished()
    assert all(r.done for r in win.runs)
    assert '越界' in win.runs[0].status
    win.export_to(root/'学习率发散')
    # Compare the explicitly implemented iterations with the original demo.
    from types import ModuleType
    deps=Path(__file__).resolve().parents[2]/'software'/'deps'
    package=ModuleType('nndesigndemos')
    package.__path__=[str(deps/'nndesigndemos')]
    sys.modules['nndesigndemos']=package
    import os
    os.environ['NNDESIGNDEMOS_PLAY_SOUND']='0'
    from nndesigndemos.book1.chapter9.Comparison_of_methods import ComparisonOfMethods
    from nndesigndemos.book1.chapter9.Newtons_method import NewtonsMethod
    original=ComparisonOfMethods(1,1,96)
    original.x_1,original.y_1=.8,-.25
    original.x_2,original.y_2=.8,-.25
    original.i=0
    ours_sd=Run(OBJECTIVES['rotated'],'sd',[.8,-.25])
    ours_cg=Run(OBJECTIVES['rotated'],'cg',[.8,-.25])
    for idx in range(2):
        original.on_animate_1(idx);original.on_animate_2(idx)
        ours_sd.step();ours_cg.step()
        np.testing.assert_allclose(ours_sd.x,[original.x_1,original.y_1],atol=1e-14)
        np.testing.assert_allclose(ours_cg.x,[original.x_2,original.y_2],atol=1e-14)
    old_newton=NewtonsMethod(1,1,96)
    old_newton.x,old_newton.y=1.5,0
    ours_newton=Run(OBJECTIVES['nonconvex'],'newton',[1.5,0])
    for idx in range(4):
        old_newton.on_animate_1(idx);ours_newton.step()
        np.testing.assert_allclose(ours_newton.x,[old_newton.x,old_newton.y],atol=1e-14)
    original.close();old_newton.close()
    print('PASS: step, timer run/pause/resume, reset, canvas click, parameter reset, export, saddle, divergence, original-demo comparison')
finally:
    win.close()
