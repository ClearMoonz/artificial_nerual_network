"""Exercise 4.5: one hidden layer, full-batch BP, no momentum."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    inputs: int = 2
    hidden: int = 4
    outputs: int = 1
    learning_rate: float = 0.8
    activation: str = 'unipolar'
    threshold: float = 0.05
    max_epochs: int = 20000
    seed: int = 42

    def validate(self):
        for name in ('inputs', 'hidden', 'outputs', 'max_epochs'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} 必须是正整数')
        if self.activation not in ('unipolar', 'bipolar'):
            raise ValueError('激活函数必须是 unipolar 或 bipolar')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('学习率必须是有限正数')
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('误差阈值必须是有限正数')
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('随机种子必须是非负整数')


def xor_data(activation='unipolar'):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    if activation == 'bipolar':
        x, y = 2 * x - 1, 2 * y - 1
    return x, y


def parse_data(text, inputs, outputs):
    """Rows contain input columns followed by target columns, no header."""
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        try:
            row = [float(v) for v in line.replace(',', ' ').replace('，', ' ').split()]
        except ValueError as exc:
            raise ValueError(f'第 {line_no} 行含非数字内容') from exc
        if len(row) != inputs + outputs:
            raise ValueError(f'第 {line_no} 行需要 {inputs + outputs} 列：输入在前，目标在后')
        rows.append(row)
    if not rows:
        raise ValueError('训练数据不能为空')
    data = np.array(rows)
    return data[:, :inputs], data[:, inputs:]


class BatchBP:
    def __init__(self, config):
        config.validate()
        self.config = config
        rng = np.random.default_rng(config.seed)
        # Last row contains the bias weights (constant input +1).
        self.V = rng.uniform(-1, 1, (config.inputs + 1, config.hidden))
        self.W = rng.uniform(-1, 1, (config.hidden + 1, config.outputs))
        self.history = []
        self.status = '未训练'

    def activate(self, z):
        # Stable logistic, including its bipolar form; no clipping of gradients.
        s = np.exp(-np.abs(z))
        logistic = np.where(z >= 0, 1 / (1 + s), s / (1 + s))
        return logistic if self.config.activation == 'unipolar' else 2 * logistic - 1

    def derivative(self, y):
        return y * (1 - y) if self.config.activation == 'unipolar' else (1 - y * y) / 2

    @staticmethod
    def augment(x):
        return np.column_stack((x, np.ones(len(x))))

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.config.inputs or not np.isfinite(x).all():
            raise ValueError('输入必须为有限数值矩阵，列数应与输入节点数一致')
        hidden = self.activate(self.augment(x) @ self.V)
        output = self.activate(self.augment(hidden) @ self.W)
        return hidden, output

    def predict(self, x):
        return self.forward(x)[1]

    def validate_data(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.forward(x)
        if len(x) == 0 or y.shape != (len(x), self.config.outputs):
            raise ValueError('样本数不能为零，目标列数必须与输出节点数一致')
        low = 0 if self.config.activation == 'unipolar' else -1
        if not np.isfinite(y).all() or np.any(y < low) or np.any(y > 1):
            raise ValueError(f'目标值必须在 [{low}, 1] 范围内')
        return x, y

    def gradients(self, x, y):
        hidden, output = self.forward(x)
        delta_out = (output - y) * self.derivative(output)
        # Both gradients use the SAME old W before either matrix is updated.
        delta_hidden = (delta_out @ self.W[:-1].T) * self.derivative(hidden)
        return self.augment(x).T @ delta_hidden, self.augment(hidden).T @ delta_out

    def metrics(self, x, y, epoch):
        residual = self.predict(x) - y
        return {'epoch': epoch, 'error': float(0.5 * np.sum(residual ** 2)),
                'rmse': float(np.sqrt(np.mean(residual ** 2)))}

    def fit(self, x, y, callback=None, should_stop=None):
        x, y = self.validate_data(x, y)
        self.history = [self.metrics(x, y, 0)]
        self.status = '训练中'
        if callback:
            callback(self.history[-1])
        if self.history[-1]['rmse'] <= self.config.threshold:
            self.status = '达到误差阈值'
            return self.history
        for epoch in range(1, self.config.max_epochs + 1):
            if should_stop and should_stop():
                self.status = '用户停止'
                break
            dv, dw = self.gradients(x, y)
            self.V -= self.config.learning_rate * dv
            self.W -= self.config.learning_rate * dw
            if not np.isfinite(self.V).all() or not np.isfinite(self.W).all():
                self.status = '数值异常'
                raise ValueError('权值出现非有限值，请减小学习率或缩放输入')
            record = self.metrics(x, y, epoch)
            self.history.append(record)
            if callback:
                callback(record)
            if record['rmse'] <= self.config.threshold:
                self.status = '达到误差阈值'
                break
        else:
            self.status = '达到最大训练轮数（未满足误差阈值）'
        return self.history
