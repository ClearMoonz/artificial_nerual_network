"""Launch the textbook's original chapter 9 demos and capture their widgets."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / 'deps'
sys.path.insert(0, str(DEPS))
RUNTIME = ROOT / ('anaconda_runtime' if sys.version_info[:2] == (3, 13) else 'runtime')
sys.path.insert(0, str(RUNTIME))
qt = RUNTIME / 'PyQt6' / 'Qt6'
os.environ['QT_PLUGIN_PATH'] = str(qt / 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt / 'plugins' / 'platforms')
os.environ['QT_API'] = 'pyqt6'
os.environ['NNDESIGNDEMOS_PLAY_SOUND'] = '0'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'mplconfig')
if '--capture' in sys.argv or '--check-clicks' in sys.argv:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
dll_handles = []
for folder in [qt / 'bin']:
    if folder.exists():
        dll_handles.append(os.add_dll_directory(str(folder)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFontDatabase
# The package's default entry eagerly imports every chapter. Load only chapter 9.
package = ModuleType('nndesigndemos')
package.__path__ = [str(DEPS / 'nndesigndemos')]
sys.modules['nndesigndemos'] = package
from nndesigndemos.book1.chapter9.Comparison_of_methods import ComparisonOfMethods
from nndesigndemos.book1.chapter9.Newtons_method import NewtonsMethod
from matplotlib.animation import FuncAnimation
import numpy as np


TOL = 1e-6
MAX_STEPS = 1000


def show_status(window, text):
    if not hasattr(window, 'label_progress'):
        window.make_label('label_progress', text, (535, 430, 165, 205),
                          font_name='Microsoft YaHei', font_size=12)
    window.label_progress.setText(text)
    window.label_progress.show()


def stop_animation(animation):
    if animation is not None and animation.event_source is not None:
        animation.event_source.stop()


class VisibleComparison(ComparisonOfMethods):
    def on_mouseclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        super().on_mouseclick(event)

    def run_animation(self, event):
        self.progress = {1: '等待迭代', 2: '等待迭代'}
        self.finished = {1: False, 2: False}
        self.x_data_1, self.y_data_1 = [event.xdata], [event.ydata]
        self.x_data_2, self.y_data_2 = [event.xdata], [event.ydata]
        self.x_1, self.y_1 = event.xdata, event.ydata
        self.x_2, self.y_2 = event.xdata, event.ydata
        self.i = 0
        # Full canvas redraw works inside Qt's scalable graphics proxy.
        self.ani_1 = FuncAnimation(self.figure, self.on_animate_1,
            init_func=self.animate_init_1, frames=MAX_STEPS,
            interval=self.animation_speed, repeat=False, blit=False)
        self.ani_2 = FuncAnimation(self.figure2, self.on_animate_2,
            init_func=self.animate_init_2, frames=MAX_STEPS,
            interval=self.animation_speed, repeat=False, blit=False)
        self.canvas.draw_idle()
        self.canvas2.draw_idle()

    def advance(self, which, idx):
        path = self.path_1 if which == 1 else self.path_2
        if self.finished[which]:
            return path,
        x = np.array([getattr(self, f'x_{which}'), getattr(self, f'y_{which}')])
        norm = np.linalg.norm(self.a @ x + self.b)
        if norm > TOL:
            if which == 1:
                super().on_animate_1(idx)
            else:
                super().on_animate_2(idx)
            x = np.array([getattr(self, f'x_{which}'), getattr(self, f'y_{which}')])
            norm = np.linalg.norm(self.a @ x + self.b)
        steps = len(getattr(self, f'x_data_{which}')) - 1
        state = '迭代中'
        if norm <= TOL:
            state = '已到达极小点'
            self.finished[which] = True
        elif idx == MAX_STEPS - 1:
            state = '达到步数上限'
            self.finished[which] = True
        if self.finished[which]:
            stop_animation(self.ani_1 if which == 1 else self.ani_2)
        self.progress[which] = f'{state}\n步数：{steps}\n梯度范数：{norm:.2e}'
        show_status(self, '最速下降\n' + self.progress[1] + '\n\n共轭梯度\n' + self.progress[2])
        return (self.path_1 if which == 1 else self.path_2),

    def on_animate_1(self, idx):
        return self.advance(1, idx)

    def on_animate_2(self, idx):
        return self.advance(2, idx)


class VisibleNewton(NewtonsMethod):
    def on_mouseclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        super().on_mouseclick(event)

    def run_animation(self, event):
        self.finished = False
        show_status(self, '已选择起点\n即将开始迭代')
        self.x_data, self.y_data = [self.x], [self.y]
        self.ani_1 = FuncAnimation(self.figure, self.on_animate_1,
            init_func=self.animate_init_1, frames=MAX_STEPS,
            interval=self.animation_speed, repeat=False, blit=False)
        self.ani_2 = FuncAnimation(self.figure2, self.on_animate_2,
            init_func=self.animate_init_2, frames=MAX_STEPS,
            interval=self.animation_speed, repeat=False, blit=False)
        self.canvas.draw_idle()
        self.canvas2.draw_idle()

    def on_animate_1(self, idx):
        if self.finished:
            return self.path_1,
        def gradient():
            t = self.y - self.x
            return np.array([-4*t**3 + 8*self.y - 1, 4*t**3 + 8*self.x + 1])
        try:
            norm = np.linalg.norm(gradient())
            if norm > TOL:
                super().on_animate_1(idx)
                norm = np.linalg.norm(gradient())
            state = '迭代中'
            if not np.isfinite(norm) or max(abs(self.x), abs(self.y)) > 1e6:
                state = '数值越界，已停止'
                self.finished = True
            elif norm <= TOL:
                t = 12*(self.y-self.x)**2
                eigenvalues = np.linalg.eigvalsh([[t, 8-t], [8-t, t]])
                if eigenvalues.min() > TOL:
                    state = '已到达局部极小点'
                elif eigenvalues.min() < -TOL and eigenvalues.max() > TOL:
                    state = '已到达鞍点'
                else:
                    state = '已到达驻点'
                self.finished = True
            elif idx == MAX_STEPS - 1:
                state = '达到步数上限'
                self.finished = True
            show_status(self, f'{state}\n步数：{len(self.x_data)-1}\n位置：\n({self.x:.6f},\n {self.y:.6f})\n梯度范数：\n{norm:.2e}')
        except (np.linalg.LinAlgError, OverflowError, FloatingPointError):
            show_status(self, '数值计算失败\n已停止，请重新选点')
            self.finished = True
        if self.finished:
            self.path_2.set_data(self.x_data[:2], self.y_data[:2])
            self.canvas2.draw_idle()
            stop_animation(self.ani_1)
            stop_animation(self.ani_2)
        return self.path_1,

app = QApplication([])
for font_path in ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/times.ttf', 'C:/Windows/Fonts/msyh.ttc']:
    QFontDatabase.addApplicationFont(font_path)
app.setStyle('Fusion')
windows = [VisibleComparison(1, 1, 96), VisibleNewton(1, 1, 96)]
for window in windows:
    window.animation_speed = 500
    if '--check-clicks' in sys.argv:
        window.animation_speed = 100
    window.show()

def start():
    windows[0].on_mouseclick(SimpleNamespace(xdata=0.8, ydata=-0.25))
    windows[1].on_mouseclick(SimpleNamespace(xdata=1.5, ydata=0.0))

if '--capture' in sys.argv:
    QTimer.singleShot(250, start)

if '--check-clicks' in sys.argv:
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import QPoint, QPointF, Qt
    def click_plot(window, canvas, axes, x, y):
        px, py = axes.transData.transform((x, y))
        point = QPoint(round(px / canvas.devicePixelRatioF()),
                       round(canvas.height() - py / canvas.devicePixelRatioF()))
        scene_point = canvas.mapTo(window, point)
        viewport_point = window._scaler.mapFromScene(QPointF(scene_point))
        QTest.mouseClick(window._scaler.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, viewport_point)
        QTest.qWait(100)
    def check_clicks():
        try:
            comparison, newton = windows
            click_plot(comparison, comparison.canvas, comparison.axes_1, 0.8, -0.25)
            QTest.qWait(1200)
            assert len(comparison.x_data_1) >= 3, 'Top plot did not animate'
            assert len(comparison.x_data_2) == 3, 'CG did not finish two steps'
            assert comparison.finished[2], 'CG failed to stop at the minimum'
            QTest.qWait(1500)
            assert comparison.finished[1], 'SD failed to run until convergence'
            assert len(comparison.x_data_1) > 6, 'SD still stops after five steps'
            assert not comparison.path_1.get_animated(), 'Path still uses blitting'
            click_plot(comparison, comparison.canvas2, comparison.axes_2, -1.2, 0.7)
            QTest.qWait(700)
            assert abs(comparison.x_data_1[0] + 1.2) < 0.04, 'Bottom plot failed to restart'
            assert len(comparison.x_data_1) >= 2, 'Restart did not animate'
            click_plot(newton, newton.canvas, newton.axes_1, 1.5, 0.0)
            QTest.qWait(600)
            assert len(newton.x_data) >= 3, 'Newton click did not animate'
            assert not newton.path_1.get_animated(), 'Newton path still uses blitting'
            assert newton.finished, 'Newton failed to stop at its stationary point'
            assert '极小点' in newton.label_progress.text(), 'Minimum status missing'
            stable_length = len(newton.x_data)
            QTest.qWait(250)
            assert len(newton.x_data) == stable_length, 'Newton keeps running after convergence'
            click_plot(newton, newton.canvas, newton.axes_1, 0.75, 0.75)
            QTest.qWait(800)
            assert newton.finished and '鞍点' in newton.label_progress.text(), 'Saddle status missing'
            comparison.on_mouseclick(SimpleNamespace(xdata=0.0, ydata=0.0))
            QTest.qWait(300)
            assert all(comparison.finished.values()), 'Zero-gradient start did not stop safely'
            # Outside the axes must be harmless.
            comparison.on_mouseclick(SimpleNamespace(xdata=None, ydata=None))
            newton.on_mouseclick(SimpleNamespace(xdata=None, ydata=None))
            print('PASS: clicks, restart, SD/CG convergence, Newton minimum/saddle, zero-gradient start', flush=True)
            app.exit(0)
        except Exception:
            import traceback
            traceback.print_exc()
            app.exit(1)
    QTimer.singleShot(300, check_clicks)

if '--capture' in sys.argv:
    out = ROOT.parent / 'results'
    out.mkdir(exist_ok=True)
    from PyQt6.QtCore import QBuffer, QIODevice
    from PIL import Image
    from io import BytesIO
    frames = []
    def capture_frame():
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        windows[0].grab().save(buffer, 'PNG')
        frames.append(Image.open(BytesIO(bytes(buffer.data()))).convert('RGB'))
    frame_timer = QTimer()
    frame_timer.timeout.connect(capture_frame)
    frame_timer.start(250)
    def capture():
        frame_timer.stop()
        # Refresh existing result artists after the original blitted animation.
        newton = windows[1]
        newton.path_1.set_data(newton.x_data, newton.y_data)
        newton.path_2.set_data(newton.x_data[:2], newton.y_data[:2])
        newton.path_1.set_animated(False)
        newton.path_2.set_animated(False)
        for window, name in zip(windows, ['comparison', 'newton']):
            window.canvas.draw()
            window.canvas2.draw()
            window.grab().save(str(out / (name + '.png')))
        if frames:
            frames[0].save(out / 'comparison.gif', save_all=True,
                           append_images=frames[1:], duration=250, loop=0)
        print('Captured original demo windows', flush=True)
        print('SD:', list(zip(windows[0].x_data_1, windows[0].y_data_1)), flush=True)
        print('CG:', list(zip(windows[0].x_data_2, windows[0].y_data_2)), flush=True)
        print('Newton:', list(zip(windows[1].x_data, windows[1].y_data))[:5], flush=True)
        app.quit()
    QTimer.singleShot(6000, capture)

sys.exit(app.exec())
