"""Chinese section 9.2 teaching interface using PyQt6 and Matplotlib."""
from datetime import datetime
from html import escape
from pathlib import Path
import json
import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPalette, QColor, QIcon, QPainter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from objectives import OBJECTIVES
from optimizers import Run, Settings, NAMES, export_runs
from surface_view import SurfacePlot, DEFAULT_CAMERA

COLORS = {'sd': '#2563eb', 'newton': '#e67722', 'cg': '#15977e'}


class ReadableToolbar(NavigationToolbar2QT):
    """Give raster toolbar icons explicit colors independent of OS theme."""
    def _icon(self, name):
        source = super()._icon(name).pixmap(QtCore.QSize(48, 48))
        icon = QIcon()
        for mode in [QIcon.Mode.Normal, QIcon.Mode.Active,
                     QIcon.Mode.Selected, QIcon.Mode.Disabled]:
            for state in [QIcon.State.Off, QIcon.State.On]:
                color = '#788596' if mode == QIcon.Mode.Disabled else (
                    '#1d4ed8' if state == QIcon.State.On or mode != QIcon.Mode.Normal else '#243449')
                pixmap = source.copy()
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), QColor(color))
                painter.end()
                icon.addPixmap(pixmap, mode, state)
        return icon


class LargeSurfaceWindow(QtWidgets.QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner, self.surface, self.scene_key = owner, None, None
        self.setWindowTitle('三维演示 · 曲面与同步迭代轨迹')
        self.resize(1080, 830)
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel('左键拖动旋转 · 右键拖动缩放 · 中键拖动平移 · 播放与主窗口同步')
        layout.addWidget(hint)
        self.figure = Figure(figsize=(10,7), layout='constrained', facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = ReadableToolbar(self.canvas, self)
        for name in ['pan','zoom']:
            self.toolbar._actions[name].setVisible(False)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas,1)
        controls = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for key, text, callback in [('step','单步',owner.step_once),('run','运行 / 继续',owner.play),
                               ('pause','暂停',owner.pause),('reset','重置 · 默认起点',owner.reset_default_start),
                               ('camera','重置视角',self.reset_camera),
                               ('export','导出结果',owner.export_dialog)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            self.buttons[key] = button
            controls.addWidget(button)
        self.progress = QtWidgets.QLabel()
        self.progress.setWordWrap(True)
        layout.addLayout(controls)
        layout.addWidget(self.progress)

    def sync(self):
        owner = self.owner
        key = (owner.objective.key, tuple(owner.plot_bounds), owner.focus_surface.isChecked(),
               tuple((run.method,tuple(run.history[0]['x'])) for run in owner.runs))
        if key != self.scene_key:
            camera = self.surface.camera if self.surface else DEFAULT_CAMERA
            self.figure.clear()
            axes = self.figure.add_subplot(projection='3d')
            self.surface = SurfacePlot(axes, owner.objective, owner.plot_bounds,
                owner.runs, owner.focus_surface.isChecked(), camera)
            self.toolbar.update()
            self.scene_key = key
        else:
            self.surface.update(owner.runs)
        self.progress.setText('   |   '.join(
            f'{NAMES[r.method]}：第 {len(r.records)} 步 · {r.status}' for r in owner.runs))
        self.canvas.draw_idle()

    def reset_camera(self):
        if self.surface:
            self.surface.reset_camera()
            self.canvas.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Pin both foreground and background so Windows dark mode cannot
        # combine light system text with the panel's light backgrounds.
        palette = QPalette()
        for role, color in {
            'Window': '#f5f7fa', 'WindowText': '#243449',
            'Base': '#ffffff', 'AlternateBase': '#edf3f9', 'Text': '#243449',
            'Button': '#e9f1fc', 'ButtonText': '#173b65',
            'ToolTipBase': '#ffffff', 'ToolTipText': '#243449',
            'Highlight': '#2563eb', 'HighlightedText': '#ffffff',
            'Link': '#2563eb', 'Light': '#ffffff', 'Mid': '#c8d6e4', 'Dark': '#64748b'
        }.items():
            palette.setColor(getattr(QPalette.ColorRole, role), QColor(color))
        for role in [QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                     QPalette.ColorRole.WindowText]:
            palette.setColor(QPalette.ColorGroup.Disabled, role, QColor('#788596'))
        QtWidgets.QApplication.instance().setPalette(palette)
        self.setWindowTitle('梯度下降实验室 · 教材第 9 章 9.2 节')
        self.resize(1390, 950)
        self.runs, self.changing, self.fast = [], True, False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.model_key = None
        self.surface, self.large_surface = None, None
        self.surface_camera = DEFAULT_CAMERA
        body = QtWidgets.QWidget()
        self.setCentralWidget(body)
        layout = QtWidgets.QHBoxLayout(body)
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(285)
        form = QtWidgets.QVBoxLayout(sidebar)
        heading = QtWidgets.QLabel('梯度下降实验室')
        heading.setStyleSheet('font-size:21px; font-weight:700; color:#15395b')
        form.addWidget(heading)
        caption = QtWidgets.QLabel('从起点出发，逐步观察搜索方向与收敛。')
        caption.setWordWrap(True)
        form.addWidget(caption)
        self.function = QtWidgets.QComboBox()
        for key, obj in OBJECTIVES.items():
            self.function.addItem(obj.name, key)
        self.function.setCurrentIndex(1)
        self.formula = QtWidgets.QLabel()
        self.formula.setWordWrap(True)
        self.algorithm = QtWidgets.QComboBox()
        for key, label in [('all', '三种算法同时对比'), *NAMES.items()]:
            self.algorithm.addItem(label, key)
        self.start1 = self.number(-100, 100, .8, 6)
        self.start2 = self.number(-100, 100, -.25, 6)
        self.mode = QtWidgets.QComboBox()
        for label, key in [('精确直线最小化（二次函数）', 'exact'),
                           ('固定学习率', 'fixed'), ('回溯线搜索（扩展）', 'backtrack')]:
            self.mode.addItem(label, key)
        self.lr = self.number(.000001, 10, .01, 6)
        self.tolerance = self.number(1e-12, 1, 1e-6, 12)
        self.maximum = QtWidgets.QSpinBox()
        self.maximum.setRange(1, 20000)
        self.maximum.setValue(1000)
        self.delay = QtWidgets.QSpinBox()
        self.delay.setRange(30, 3000)
        self.delay.setSingleStep(100)
        self.delay.setValue(500)
        self.delay.setSuffix(' ms / 步')
        grid = QtWidgets.QFormLayout()
        for label, widget in [('教材实验', self.function), ('算法', self.algorithm),
                ('起点 x1', self.start1), ('起点 x2', self.start2),
                ('最速下降步长', self.mode), ('固定学习率 α', self.lr),
                ('梯度停止阈值', self.tolerance), ('最大迭代步数', self.maximum),
                ('动画间隔', self.delay)]:
            grid.addRow(label, widget)
        form.addLayout(grid)
        form.addWidget(self.formula)
        self.presets = QtWidgets.QComboBox()
        form.addWidget(QtWidgets.QLabel('教材参数快捷选择'))
        form.addWidget(self.presets)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet('background:#eaf1f8;padding:10px;color:#244564;')
        form.addWidget(self.hint)
        controls = QtWidgets.QGridLayout()
        self.buttons = {}
        for idx, (key, label, fn) in enumerate([
            ('step', '单步', self.step_once), ('run', '运行 / 继续', self.play),
            ('pause', '暂停', self.pause), ('reset', '重置 · 默认起点', self.reset_default_start),
            ('finish', '计算至结束', self.finish), ('export', '导出结果', self.export_dialog)]):
            button = QtWidgets.QPushButton(label)
            button.setObjectName(key + 'Button')
            button.clicked.connect(fn)
            self.buttons[key] = button
            controls.addWidget(button, idx//2, idx%2)
        form.addLayout(controls)
        note = QtWidgets.QLabel('点击左上等高线图选择起点，再点“运行”。\n修改计算参数会重置结果；动画速度仅影响播放。')
        note.setWordWrap(True)
        form.addWidget(note)
        form.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(306)
        layout.addWidget(scroll)
        right = QtWidgets.QVBoxLayout()
        self.figure = Figure(figsize=(10, 6.4), layout='constrained', facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = ReadableToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        view_controls = QtWidgets.QHBoxLayout()
        view_controls.addWidget(QtWidgets.QLabel('视图'))
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItem('二维 · 牛顿局部模型', '2d')
        self.view_mode.addItem('三维曲面 · 同步轨迹', '3d')
        view_controls.addWidget(self.view_mode)
        self.focus_surface = QtWidgets.QCheckBox('聚焦谷底')
        self.focus_surface.setChecked(True)
        self.focus_surface.setToolTip('非凸函数隐藏 F>12 的高处，便于看清谷底与鞍点；不改变计算结果。')
        view_controls.addWidget(self.focus_surface)
        self.large_surface_button = QtWidgets.QPushButton('放大三维')
        self.large_surface_button.clicked.connect(self.open_large_surface)
        view_controls.addWidget(self.large_surface_button)
        self.camera_button = QtWidgets.QPushButton('重置视角')
        self.camera_button.clicked.connect(self.reset_surface_camera)
        view_controls.addWidget(self.camera_button)
        view_controls.addStretch()
        right.addLayout(view_controls)
        right.addWidget(self.canvas, 5)
        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(['算法', '步数', '状态', 'F(x)', '‖梯度‖',
                                              '函数调用', '梯度调用', 'Hessian调用', '计算 ms'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(132)
        right.addWidget(self.table)
        self.detail = QtWidgets.QTextBrowser()
        self.detail.setMinimumHeight(100)
        self.detail.setMaximumHeight(170)
        right.addWidget(self.detail, 1)
        layout.addLayout(right, 1)
        self.setStyleSheet('QMainWindow,QWidget{font-family:"Microsoft YaHei";font-size:12px;color:#243449;}'
            'QMainWindow{background:#f5f7fa;} QPushButton{padding:7px;border:1px solid #b7cbe2;'
            'border-radius:5px;background:#e9f1fc;color:#173b65;font-weight:600;}'
            'QPushButton:hover{background:#d7e6fa;color:#143960;}'
            'QPushButton:pressed{background:#bfd5f1;color:#143960;}'
            'QPushButton:focus{border:2px solid #2563eb;}'
            'QPushButton#runButton{background:#2563eb;color:white;border-color:#2563eb;}'
            'QPushButton#runButton:hover{background:#1d4ed8;color:white;}'
            'QPushButton#runButton:pressed{background:#1e40af;color:white;}'
            'QPushButton:disabled{background:#edf1f5;color:#788596;border-color:#d6dfe9;}'
            'QComboBox,QDoubleSpinBox,QSpinBox{padding:3px;background:white;color:#243449;'
            'border:1px solid #c8d6e4;border-radius:3px;}'
            'QComboBox:disabled,QDoubleSpinBox:disabled,QSpinBox:disabled{background:#edf1f5;color:#788596;}'
            'QComboBox QAbstractItemView{background:white;color:#243449;selection-background-color:#2563eb;'
            'selection-color:white;}'
            'QTextBrowser,QTableWidget{background:white;color:#243449;}'
            'QToolBar{background:#f5f7fa;border:none;}'
            'QToolButton{background:transparent;color:#243449;padding:5px;'
            'border:1px solid transparent;border-radius:4px;}'
            'QToolButton:hover{background:#d7e6fa;border-color:#b7cbe2;}'
            'QToolButton:pressed,QToolButton:checked{background:#c9dcfb;border-color:#2563eb;}'
            'QToolButton:disabled{color:#788596;}'
            'QHeaderView::section{background:#eaf1f8;color:#244564;padding:4px;border:1px solid #d4dee9;}'
            'QToolTip{background:white;color:#243449;border:1px solid #c8d6e4;}')
        self.statusBar().showMessage('选择教材实验，再运行或单步。')
        self.canvas.mpl_connect('button_press_event', self.choose_start)
        self.function.currentIndexChanged.connect(self.change_objective)
        self.algorithm.currentIndexChanged.connect(self.parameter_changed)
        self.mode.currentIndexChanged.connect(self.parameter_changed)
        for widget in [self.start1, self.start2, self.lr, self.tolerance, self.maximum]:
            widget.valueChanged.connect(self.parameter_changed)
        self.presets.currentIndexChanged.connect(self.use_preset)
        self.delay.valueChanged.connect(self.change_speed)
        self.view_mode.currentIndexChanged.connect(self.change_view)
        self.focus_surface.toggled.connect(self.change_view)
        self.change_objective()
        self.changing = False
        self.reset()

    @staticmethod
    def number(low, high, value, decimals):
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(low, high)
        box.setValue(value)
        box.setSingleStep(.01 if decimals < 10 else 1e-6)
        return box

    @property
    def objective(self):
        return OBJECTIVES[self.function.currentData()]

    def change_objective(self):
        self.changing = True
        self.timer.stop()
        obj = self.objective
        nonconvex = not obj.quadratic
        self.algorithm.model().item(0).setEnabled(not nonconvex)
        self.algorithm.model().item(3).setEnabled(not nonconvex)
        self.mode.model().item(0).setEnabled(not nonconvex)
        self.algorithm.setCurrentIndex(2 if nonconvex else 0)
        self.mode.setCurrentIndex(1 if obj.key == 'elongated' or nonconvex else 0)
        self.start1.setValue(obj.default_start[0])
        self.start2.setValue(obj.default_start[1])
        self.lr.setValue(.01)
        self.formula.setText(obj.formula)
        self.presets.clear()
        self.presets.addItem('自定义参数', None)
        if obj.key == 'elongated':
            for a, text in [(.01,'图9.1 · 平稳收敛'),(.035,'图9.2 · 衰减振荡'),
                            (.039,'图9.3 · 稳定'),(.04,'边界 · 不收敛'),(.041,'图9.3 · 发散')]:
                self.presets.addItem(f'α={a} · {text}', ('rate', a))
        elif nonconvex:
            for point, text in [((1.5,0),'图9.6 · 全局极小点'),((-1.5,0),'图9.7 · 局部极小点'),
                               ((.75,.75),'图9.8 · 鞍点'),((1.15,.75),'图9.9 · 局部极小点')]:
                self.presets.addItem(text, ('start', point))
        else:
            self.presets.addItem('教材起点 (0.8, −0.25)', ('start', (.8,-.25)))
        self.changing = False
        self.reset()

    def use_preset(self):
        data = self.presets.currentData()
        if self.changing or data is None:
            return
        self.changing = True
        if data[0] == 'rate':
            self.mode.setCurrentIndex(1)
            self.lr.setValue(data[1])
            self.start1.setValue(.5)
            self.start2.setValue(.5)
        else:
            self.start1.setValue(data[1][0])
            self.start2.setValue(data[1][1])
        self.changing = False
        self.reset()

    def parameter_changed(self):
        if not self.changing:
            self.presets.blockSignals(True)
            self.presets.setCurrentIndex(0)
            self.presets.blockSignals(False)
            self.reset()

    def change_speed(self):
        if self.timer.isActive() and not self.fast:
            self.timer.setInterval(self.delay.value())

    @property
    def is_3d(self):
        return self.view_mode.currentData() == '3d'

    def change_view(self):
        if self.changing:
            return
        if self.surface:
            self.surface_camera = self.surface.camera
        for method in ['pan','zoom']:
            if self.toolbar._actions[method].isChecked():
                self.toolbar._actions[method].trigger()
        self.build_plots()
        self.render()
        self.statusBar().showMessage('视图已切换，迭代结果保留。三维图左键拖动旋转，选起点使用左侧二维图。')

    def reset_surface_camera(self):
        self.surface_camera = DEFAULT_CAMERA
        if self.surface:
            self.surface.reset_camera()
            self.canvas.draw_idle()
        if self.large_surface:
            self.large_surface.reset_camera()

    def open_large_surface(self):
        if not self.is_3d:
            self.view_mode.setCurrentIndex(1)
        if self.large_surface is None:
            self.large_surface = LargeSurfaceWindow(self)
        self.large_surface.sync()
        self.large_surface.show()
        self.large_surface.raise_()

    def reset(self):
        self.timer.stop()
        self.fast = False
        selected = self.algorithm.currentData()
        methods = list(NAMES) if selected == 'all' else [selected]
        settings = Settings(self.mode.currentData(), self.lr.value(),
                            self.tolerance.value(), self.maximum.value())
        try:
            self.runs = [Run(self.objective, method,
                [self.start1.value(), self.start2.value()], settings) for method in methods]
        except (ValueError, FloatingPointError) as exc:
            self.runs = []
            self.statusBar().showMessage(str(exc))
        applicable = selected in ('all', 'sd')
        self.mode.setEnabled(applicable)
        self.lr.setEnabled(applicable and self.mode.currentData() == 'fixed')
        if self.objective.quadratic:
            bound = 2/np.linalg.eigvalsh(self.objective.matrix).max()
            hint = f'正定二次函数固定步长稳定范围：\n0 < α < {bound:.6g}\n'
            if self.mode.currentData() == 'fixed' and applicable and self.lr.value() >= bound:
                hint += '当前 α 不满足稳定条件，可观察振荡或发散。\n'
            hint += '共轭梯度始终使用精确线搜索；牛顿步长始终为 1。'
        else:
            hint = '牛顿法可能到达鞍点，也可能不下降。\n本实验不启用共轭梯度。\n最速下降可选固定步长或回溯；回溯是教材之外的扩展。'
        self.hint.setText(hint)
        self.build_plots()
        self.render()
        if self.runs:
            self.statusBar().showMessage(
                f'已回到起点 ({self.start1.value():.6g}, {self.start2.value():.6g})，步数归零、轨迹清空。可单步或重新运行。')

    def reset_default_start(self):
        """User reset restores the textbook point; internal resets keep selected points."""
        self.timer.stop()
        self.changing = True
        self.start1.setValue(self.objective.default_start[0])
        self.start2.setValue(self.objective.default_start[1])
        # Clear the old shortcut label, which may refer to another starting point.
        self.presets.setCurrentIndex(0)
        self.changing = False
        self.reset()
        self.statusBar().showMessage(
            f'已恢复教材默认起点 ({self.start1.value():.6g}, {self.start2.value():.6g})，轨迹清空；其他参数保留。')

    def build_plots(self):
        if self.surface:
            self.surface_camera = self.surface.camera
        self.figure.clear()
        grid = self.figure.add_gridspec(2,2)
        self.path_axes = self.figure.add_subplot(grid[0,0])
        self.model_axes = self.figure.add_subplot(grid[0,1], projection='3d' if self.is_3d else None)
        self.f_axes = self.figure.add_subplot(grid[1,0])
        self.g_axes = self.figure.add_subplot(grid[1,1])
        self.surface = None
        obj = self.objective
        xmin, xmax, ymin, ymax = obj.bounds
        # Keep all starts visible; use the same axes for all algorithms.
        x0, y0 = self.start1.value(), self.start2.value()
        xmin, xmax = min(xmin, x0-.2), max(xmax, x0+.2)
        ymin, ymax = min(ymin, y0-.2), max(ymax, y0+.2)
        self.plot_bounds = (xmin,xmax,ymin,ymax)
        self.mesh = np.meshgrid(np.linspace(xmin, xmax, 130), np.linspace(ymin, ymax, 130))
        X, Y = self.mesh
        if obj.quadratic:
            levels = np.geomspace(.002, max(float(obj.grid(X,Y).max()), 1), 14)
        else:
            levels = [-.5,0,.5,1,2,3,4,6,8,12,20,35]
        self.path_axes.contour(X, Y, obj.grid(X,Y), levels=levels, cmap='Blues', linewidths=.8)
        self.path_axes.plot(x0,y0,'*',color='#263a50',markersize=10,label='起点')
        self.path_axes.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x1',ylabel='x2',title='等高线与迭代轨迹 · 点击选起点')
        self.path_axes.set_aspect('equal', adjustable='box')
        self.path_lines, self.f_lines, self.g_lines, self.arrows = {}, {}, {}, []
        for run in self.runs:
            label, color = NAMES[run.method], COLORS[run.method]
            self.path_lines[run.method], = self.path_axes.plot([],[],'o-',color=color,markersize=3,lw=1.7,label=label)
            self.f_lines[run.method], = self.f_axes.plot([],[],color=color,label=label)
            self.g_lines[run.method], = self.g_axes.plot([],[],color=color,label=label)
        self.path_axes.legend(fontsize=8, loc='best')
        self.f_axes.set(title='函数值 F(x_k) · 线性坐标',xlabel='迭代步数 k',ylabel='F(x_k)')
        self.g_axes.set(title='梯度范数 · 越小越接近驻点',xlabel='迭代步数 k',ylabel='梯度范数',yscale='log')
        for axes in [self.f_axes, self.g_axes]:
            axes.grid(True, alpha=.2)
        self.g_axes.axhline(self.tolerance.value(), color='#8192a1',ls='--',lw=1,label='停止阈值')
        self.model_key = None
        self.camera_button.setEnabled(self.is_3d)
        self.focus_surface.setEnabled(self.is_3d and not obj.quadratic)
        if self.is_3d:
            self.surface = SurfacePlot(self.model_axes, obj, self.plot_bounds, self.runs,
                self.focus_surface.isChecked(), self.surface_camera)
        self.toolbar.update()

    def choose_start(self, event):
        if event.inaxes != self.path_axes or event.xdata is None or self.toolbar.mode:
            return
        self.changing = True
        self.start1.setValue(event.xdata)
        self.start2.setValue(event.ydata)
        self.changing = False
        self.reset()

    def step_once(self):
        self.pause()
        for run in self.runs:
            run.step()
        self.render()
        self.show_run_status()

    def play(self):
        if self.runs and not all(r.done for r in self.runs):
            self.fast = False
            self.timer.start(self.delay.value())
            self.statusBar().showMessage('播放中。可随时暂停，或修改动画速度。')
        elif self.runs:
            self.statusBar().showMessage('本轮已结束。可重置，或在图中选择新的起点。')

    def pause(self):
        self.timer.stop()
        self.statusBar().showMessage('已暂停。可单步或继续运行。')

    def finish(self):
        if self.runs and not all(r.done for r in self.runs):
            self.fast = True
            self.timer.start(1)
            self.statusBar().showMessage('正在分批计算至结束；可暂停。')

    def tick(self):
        for _ in range(25 if self.fast else 1):
            for run in self.runs:
                run.step()
            if all(r.done for r in self.runs):
                self.timer.stop()
                break
        self.render()
        self.show_run_status()

    def show_run_status(self):
        if self.runs and all(r.done for r in self.runs):
            self.statusBar().showMessage('；'.join(f'{NAMES[r.method]}：{r.status}' for r in self.runs))

    def render(self):
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []
        for run in self.runs:
            xy = np.array([s['x'] for s in run.history])
            k = [s['k'] for s in run.history]
            self.path_lines[run.method].set_data(xy[:,0],xy[:,1])
            self.f_lines[run.method].set_data(k,[s['f'] for s in run.history])
            self.g_lines[run.method].set_data(k,[max(s['norm'],1e-16) for s in run.history])
            if run.records:
                rec = run.records[-1]
                arrow = self.path_axes.annotate('',xy=rec['x_new'],xytext=rec['x_old'],
                    arrowprops=dict(arrowstyle='->',color=COLORS[run.method],lw=1.5))
                self.arrows.append(arrow)
        for axes in [self.f_axes,self.g_axes]:
            axes.relim()
            axes.autoscale_view()
        newton = next((r for r in self.runs if r.method == 'newton'),None)
        key = len(newton.records) if newton else -1
        if self.surface:
            self.surface.update(self.runs)
        elif key != self.model_key:
            self.model_axes.clear()
            if newton:
                if newton.records:
                    rec = newton.records[-1]
                    x, g, h, f = np.array(rec['x_old']),np.array(rec['g_old']),np.array(rec['hessian']),rec['f_old']
                else:
                    x, g, f = newton.x,newton.g,newton.f
                    h = self.objective.hessian(x)
                X,Y = self.mesh
                u,v = X-x[0],Y-x[1]
                Z = f+g[0]*u+g[1]*v+.5*(h[0,0]*u*u+2*h[0,1]*u*v+h[1,1]*v*v)
                self.model_axes.contour(X,Y,Z,levels=12,cmap='Oranges',linewidths=.9)
                self.model_axes.plot(*x,'o',color=COLORS['newton'])
                if newton.records:
                    self.model_axes.annotate('',xy=newton.x,xytext=x,
                        arrowprops=dict(arrowstyle='->',color=COLORS['newton']))
                self.model_axes.set(title=f'牛顿法：第 {key} 步的局部二次近似',xlabel='x1',ylabel='x2')
                self.model_axes.set_aspect('equal',adjustable='box')
            else:
                self.model_axes.axis('off')
                self.model_axes.text(.05,.85,'当前仅展示一阶方法\n\n选择“牛顿法”或“三种算法同时对比”\n可观察局部二次近似。',
                    transform=self.model_axes.transAxes,va='top',fontsize=12,color='#52677a')
            self.model_key = key
        self.table.setRowCount(len(self.runs))
        for row,run in enumerate(self.runs):
            s=run.summary()
            vals=[s['name'],s['steps'],s['status'],f"{s['f']:.7g}",f"{s['gradient_norm']:.3e}",
                  s['counts']['value'],s['counts']['gradient'],s['counts']['hessian'],f"{s['cpu_ms']:.3f}"]
            for col,value in enumerate(vals):
                self.table.setItem(row,col,QtWidgets.QTableWidgetItem(str(value)))
        self.show_calculations()
        self.canvas.draw_idle()
        if self.large_surface and self.large_surface.isVisible():
            self.large_surface.sync()

    def show_calculations(self):
        def vec(x):
            return '('+', '.join(f'{v:.7g}' for v in x)+')'
        parts = ['<b>逐步计算</b> · 当前显示最后一次更新；内部计算保留双精度。']
        for run in self.runs:
            title = f'<b style="color:{COLORS[run.method]}">{escape(NAMES[run.method])}</b>'
            if not run.records:
                parts.append(f'{title}：起点 {vec(run.x)}，F={run.f:.7g}，{escape(run.status)}')
                continue
            rec=run.records[-1]
            text=f"{title}：x_{rec['k']}={vec(rec['x_old'])}，g={vec(rec['g_old'])}，p={vec(rec['direction'])}，α={rec['alpha']:.8g}"
            if rec['beta'] is not None:
                text+=f"，β(FR)={rec['beta']:.8g}"
            text+=f" → x_{rec['k']+1}={vec(rec['x_new'])}"
            if rec['hessian_eigenvalues'] is not None:
                text+=f"；Hessian特征值={vec(rec['hessian_eigenvalues'])}"
            if run.method=='newton':
                text+=f"；解 H·p=−g，H={escape(str(rec['hessian']))}"
            if len(run.records)>1 and self.objective.quadratic:
                old=np.array(run.records[-2]['direction'])
                current=np.array(rec['direction'])
                product=old @ (self.objective.matrix @ current) if run.method=='cg' else old @ current
                if run.method in ('sd','cg') and (run.method=='cg' or run.settings.step_mode=='exact'):
                    text+=f"；{'共轭 p前ᵀHp' if run.method=='cg' else '正交 p前ᵀp'}={product:.2e}"
            parts.append(text)
        parts.append('<span style="color:#617383">调用次数包含实际算法计算与终止判断；计算用时不含动画等待和绘图。牛顿法可能到达鞍点，梯度小不等同于全局最优。</span>')
        self.detail.setHtml('<br><br>'.join(parts))

    def export_to(self, directory):
        if not self.runs:
            raise ValueError('没有可导出的实验')
        self.pause()
        self.canvas.draw()
        export_runs(directory,self.runs,self.figure)
        if self.is_3d:
            # Also provide a dedicated large 3D image suitable for a presentation.
            figure = Figure(figsize=(9,7), layout='constrained', facecolor='white')
            camera = (self.large_surface.surface.camera
                if self.large_surface and self.large_surface.isVisible() else self.surface.camera)
            SurfacePlot(figure.add_subplot(projection='3d'), self.objective,
                        self.plot_bounds, self.runs, self.focus_surface.isChecked(), camera)
            figure.savefig(Path(directory)/'surface3d.png', dpi=160, facecolor='white')
            metadata = {'view':'3d', 'axes':['x1','x2','F(x1,x2)'],
                'camera':list(camera), 'focus_valley':self.focus_surface.isChecked(),
                'surface_high_region_hidden':self.surface.clipped, 'bounds':list(self.plot_bounds),
                'note':'仅切换可视化；三维点高度使用原始记录的函数值。聚焦谷底时隐藏F>12高处，不修改数据。'}
            (Path(directory)/'view.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        self.statusBar().showMessage(f'已导出 CSV、JSON、曲线图和界面截图：{directory}')
        self.grab().save(str(Path(directory)/'interface.png'))

    def export_dialog(self):
        root = Path(__file__).resolve().parent.parent/'results'
        root.mkdir(exist_ok=True)
        selected=QtWidgets.QFileDialog.getExistingDirectory(self,'选择导出位置',str(root))
        if not selected:
            return
        dest=Path(selected)/datetime.now().strftime('实验_%Y%m%d_%H%M%S_%f')
        try:
            self.export_to(dest)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self,'导出失败',str(exc))

    def closeEvent(self,event):
        self.timer.stop()
        if self.large_surface:
            self.large_surface.close()
        super().closeEvent(event)
