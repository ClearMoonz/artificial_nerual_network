"""Standalone Chinese teaching panel; original textbook demos remain in software/."""
import sys
from runtime import configure
configure('--capture' in sys.argv)
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtCore import QTimer
import matplotlib
from matplotlib import font_manager
from pathlib import Path

font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from main_window import MainWindow


def main():
    app=QApplication([])
    app.setStyle('Fusion')
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont('Microsoft YaHei',10))
    win=MainWindow()
    if '--3d' in sys.argv:
        win.view_mode.setCurrentIndex(1)
    win.show()
    if '--capture' in sys.argv:
        def capture():
            for run in win.runs:
                run.finish()
            win.render()
            app.processEvents()
            folder = '三维界面预览' if win.is_3d else '完整界面预览'
            win.export_to(Path(__file__).resolve().parent.parent/'results'/folder)
            app.quit()
        QTimer.singleShot(400,capture)
    return app.exec()


if __name__=='__main__':
    sys.exit(main())
