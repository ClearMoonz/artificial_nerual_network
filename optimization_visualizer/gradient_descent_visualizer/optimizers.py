"""Explicit textbook iterations, independent of Qt and plotting."""
from dataclasses import dataclass, asdict
from time import perf_counter
import csv
import json
from pathlib import Path
import numpy as np

NAMES = {'sd': '最速下降法', 'newton': '牛顿法', 'cg': '共轭梯度法（FR）'}


@dataclass(frozen=True)
class Settings:
    step_mode: str = 'exact'
    learning_rate: float = .01
    tolerance: float = 1e-6
    max_steps: int = 1000
    coordinate_limit: float = 1e6

    def validate(self):
        if self.step_mode not in ('exact', 'fixed', 'backtrack'):
            raise ValueError('未知步长模式')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('学习率必须是有限的正数')
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('梯度阈值必须是有限的正数')
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError('最大步数必须为正整数')
        if not np.isfinite(self.coordinate_limit) or self.coordinate_limit <= 0:
            raise ValueError('坐标上限必须为有限的正数')


class Run:
    def __init__(self, objective, method, start, settings=None):
        self.objective, self.method = objective, method
        self.settings = settings or Settings()
        self.settings.validate()
        if method not in NAMES:
            raise ValueError('未知算法')
        if method == 'cg' and not objective.quadratic:
            raise ValueError('本版共轭梯度法只用于正定二次函数')
        if method == 'sd' and self.settings.step_mode == 'exact' and not objective.quadratic:
            raise ValueError('非二次函数请使用固定学习率或回溯线搜索')
        if objective.quadratic and np.linalg.eigvalsh(objective.matrix).min() <= 0:
            raise ValueError('二次函数必须具有正定 Hessian')
        self.x = np.array(start, dtype=float)
        if self.x.shape != (2,) or not np.isfinite(self.x).all():
            raise ValueError('起点必须是两个有限数值')
        if abs(self.x).max() > self.settings.coordinate_limit:
            raise ValueError('起点超出数值保护范围')
        self.counts = {'value': 0, 'gradient': 0, 'hessian': 0}
        self.cpu_seconds = 0.
        self.records, self.history = [], []
        self.previous_g, self.previous_p = None, None
        self.status = '等待运行'
        self.done = False
        then = perf_counter()
        self.f = self.evaluate('value', self.x)
        self.g = self.evaluate('gradient', self.x)
        if not np.isfinite(self.f) or not np.isfinite(self.g).all():
            raise ValueError('初始点函数值或梯度不是有限数')
        self.save_state()
        self.check_stationary()
        self.cpu_seconds += perf_counter()-then

    def evaluate(self, what, x):
        self.counts[what] += 1
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return getattr(self.objective, what)(x)

    def save_state(self):
        self.history.append({'k': len(self.records), 'x': self.x.tolist(),
            'f': float(self.f), 'gradient': self.g.tolist(),
            'norm': float(np.linalg.norm(self.g))})

    def stop(self, reason):
        self.status, self.done = reason, True

    def check_stationary(self):
        if np.linalg.norm(self.g) > self.settings.tolerance:
            return False
        h = self.evaluate('hessian', self.x)
        ev = np.linalg.eigvalsh(h)
        if ev.min() > 1e-10:
            label = '已到达极小点' if self.objective.quadratic else '已到达局部极小点'
        elif ev.max() < -1e-10:
            label = '已到达局部极大点'
        elif ev.min() < -1e-10 and ev.max() > 1e-10:
            label = '已到达鞍点'
        else:
            label = '已到达驻点（类型待定）'
        self.stop(label)
        return True

    def step(self):
        if self.done:
            return False
        then = perf_counter()
        try:
            return self._step()
        except np.linalg.LinAlgError:
            self.stop('Hessian 奇异，无法求解')
        except (FloatingPointError, OverflowError):
            self.stop('非有限数值，保护停止')
        finally:
            self.cpu_seconds += perf_counter()-then
        return False

    def _step(self):
        self.status = '迭代中'
        x, g, f = self.x.copy(), self.g.copy(), self.f
        beta, h = None, None
        if self.method == 'newton':
            h = self.evaluate('hessian', x)
            if np.linalg.matrix_rank(h) < 2:
                raise np.linalg.LinAlgError('Singular Hessian at working precision')
            p = np.linalg.solve(h, -g)
            alpha = 1.
        else:
            p = -g
            if self.method == 'cg' and self.previous_g is not None:
                denominator = float(self.previous_g @ self.previous_g)
                if denominator <= np.finfo(float).tiny:
                    self.stop('共轭系数分母退化')
                    return False
                beta = float(g @ g)/denominator
                p = -g + beta*self.previous_p
            elif self.method == 'cg':
                beta = 0.
            mode = 'exact' if self.method == 'cg' else self.settings.step_mode
            if mode == 'fixed':
                alpha = self.settings.learning_rate
            elif mode == 'exact':
                h = self.evaluate('hessian', x)
                denominator = float(p @ h @ p)
                if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
                    self.stop('线搜索曲率退化，已停止')
                    return False
                alpha = -float(g @ p)/denominator
            else:
                # Explicit extension for nonquadratic steepest descent only.
                alpha = 1.
                slope = float(g @ p)
                accepted = False
                for _ in range(60):
                    trial = x+alpha*p
                    if abs(trial).max() <= self.settings.coordinate_limit:
                        trial_f = self.evaluate('value', trial)
                        if np.isfinite(trial_f) and trial_f <= f + 1e-4*alpha*slope:
                            accepted = True
                            break
                    alpha *= .5
                if not accepted:
                    self.stop('回溯线搜索失败')
                    return False
        candidate = x+alpha*p
        if not np.isfinite(candidate).all() or abs(candidate).max() > self.settings.coordinate_limit:
            self.stop('坐标越界，保护停止（可能发散）')
            return False
        new_f = self.evaluate('value', candidate)
        new_g = self.evaluate('gradient', candidate)
        if not np.isfinite(new_f) or not np.isfinite(new_g).all():
            self.stop('非有限数值，保护停止')
            return False
        rec = {'k': len(self.records), 'x_old': x.tolist(), 'f_old': float(f),
            'g_old': g.tolist(), 'direction': p.tolist(), 'alpha': float(alpha),
            'beta': beta, 'x_new': candidate.tolist(), 'f_new': float(new_f),
            'gradient_new': new_g.tolist(), 'norm_new': float(np.linalg.norm(new_g)),
            'hessian': h.tolist() if h is not None else None,
            'hessian_eigenvalues': np.linalg.eigvalsh(h).tolist() if h is not None else None}
        self.records.append(rec)
        self.previous_g, self.previous_p = g, p.copy()
        self.x, self.f, self.g = candidate, new_f, new_g
        self.save_state()
        if self.check_stationary():
            return True
        scale = max(1., float(np.linalg.norm(x)))
        if np.linalg.norm(candidate-x) <= 4*np.finfo(float).eps*scale:
            self.stop('步长过小，停滞（梯度未达标）')
        elif len(self.records) >= self.settings.max_steps:
            self.stop('达到最大步数（梯度未达标）')
        return True

    def finish(self):
        while not self.done:
            self.step()
        return self

    def summary(self):
        return {'algorithm': self.method, 'name': NAMES[self.method],
            'steps': len(self.records), 'status': self.status,
            'x': self.x.tolist(), 'f': float(self.f),
            'gradient_norm': float(np.linalg.norm(self.g)),
            'counts': self.counts.copy(), 'cpu_ms': self.cpu_seconds*1000}


def export_runs(directory, runs, figure=None):
    """Export all states (including k=0), transitions, and reproducible settings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ['algorithm', 'k', 'x1', 'x2', 'f', 'g1', 'g2', 'gradient_norm',
              'alpha_to_next', 'beta_to_next', 'p1_to_next', 'p2_to_next', 'status']
    with (directory/'iterations.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for state in run.history:
                k = state['k']
                rec = run.records[k] if k < len(run.records) else None
                writer.writerow(dict(zip(fields, [NAMES[run.method], k, *state['x'],
                    state['f'], *state['gradient'], state['norm'],
                    rec['alpha'] if rec else '', rec['beta'] if rec and rec['beta'] is not None else '',
                    *(rec['direction'] if rec else ['', '']),
                    run.status if k == len(run.history)-1 else ''])) )
    payload = {'objective': runs[0].objective.key, 'formula': runs[0].objective.formula,
        'note': 'CPU时间包含算法计算及驻点判断，不含播放等待、绘图或导出；调用次数为实际调用。',
        'runs': [{'settings': asdict(r.settings), 'summary': r.summary(),
                  'history': r.history, 'transitions': r.records} for r in runs]}
    (directory/'experiment.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    if figure is not None:
        figure.savefig(directory/'plots.png', dpi=160, facecolor='white')
    return directory
