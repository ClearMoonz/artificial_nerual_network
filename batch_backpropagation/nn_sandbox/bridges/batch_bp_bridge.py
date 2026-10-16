"""Qt bridge: training in a worker, UI updates by queued signals."""
import threading
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, pyqtProperty
from nn_sandbox.backend.algorithms.batch_bp import BatchBP, Config, parse_data


class Worker(QThread):
    progress = pyqtSignal(int, float)
    result = pyqtSignal(str)

    def __init__(self, config, x, y):
        super().__init__()
        self.config, self.x, self.y = config, x, y
        self.cancel = threading.Event()

    def run(self):
        try:
            model = BatchBP(self.config)
            def update(record):
                if record['epoch'] % 20 == 0 or record['rmse'] <= self.config.threshold:
                    self.progress.emit(record['epoch'], record['rmse'])
            model.fit(self.x, self.y, update, self.cancel.is_set)
            last = model.history[-1]
            self.progress.emit(last['epoch'], last['rmse'])
            lines = [f"{model.status}；轮数 {last['epoch']}；RMSE={last['rmse']:.6f}；E={last['error']:.6f}"]
            for x, target, output in zip(self.x, self.y, model.predict(self.x)):
                lines.append(f'输入 {x.tolist()} → 输出 {output.round(6).tolist()}；目标 {target.tolist()}')
            self.result.emit('\n'.join(lines))
        except Exception as exc:
            self.result.emit(f'训练失败：{exc}')


class BatchBPBridge(QObject):
    changed = pyqtSignal()
    point = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self._busy = False
        self._message = '请设置参数并开始训练。每轮对全部样本统一更新一次权值。'
        self.worker = None

    @pyqtProperty(bool, notify=changed)
    def busy(self):
        return self._busy

    @pyqtProperty(str, notify=changed)
    def message(self):
        return self._message

    @pyqtSlot(int, int, int, str, str, str, str, str, str)
    def start(self, inputs, hidden, outputs, lr, activation, threshold, epochs, seed, data):
        if self._busy:
            return
        try:
            cfg = Config(inputs, hidden, outputs, float(lr), activation,
                         float(threshold), int(epochs), int(seed))
            cfg.validate()
            x, y = parse_data(data, inputs, outputs)
            BatchBP(cfg).validate_data(x, y)
        except Exception as exc:
            self._message = f'参数或数据错误：{exc}'
            self.changed.emit()
            return
        self._busy = True
        self._message = '训练中……'
        self.changed.emit()
        self.worker = Worker(cfg, x, y)
        self.worker.progress.connect(self.point)
        self.worker.result.connect(self.receive_result)
        self.worker.finished.connect(self.done)
        self.worker.start()

    @pyqtSlot(str)
    def receive_result(self, text):
        self._message = text
        self.changed.emit()

    @pyqtSlot()
    def done(self):
        self._busy = False
        self.changed.emit()

    @pyqtSlot()
    def stop(self):
        if self.worker:
            self.worker.cancel.set()

    def shutdown(self):
        self.stop()
        if self.worker:
            self.worker.wait()
