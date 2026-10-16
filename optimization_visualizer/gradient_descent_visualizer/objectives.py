"""The three two-variable objectives from Neural Network Design, section 9.2."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Objective:
    key: str
    name: str
    formula: str
    default_start: tuple
    bounds: tuple
    matrix: object = None

    @property
    def quadratic(self):
        return self.matrix is not None

    def value(self, x):
        x = np.asarray(x, dtype=float)
        if self.quadratic:
            return float(0.5 * x @ self.matrix @ x)
        u, v = x
        return float((v-u)**4 + 8*u*v-u+v+3)

    def grid(self, u, v):
        if self.quadratic:
            a = self.matrix
            return (a[0, 0]*u*u + 2*a[0, 1]*u*v + a[1, 1]*v*v)/2
        return (v-u)**4 + 8*u*v-u+v+3

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        if self.quadratic:
            return self.matrix @ x
        u, v = x
        t = v-u
        return np.array([-4*t**3 + 8*v-1, 4*t**3 + 8*u+1])

    def hessian(self, x):
        if self.quadratic:
            return self.matrix.copy()
        t = 12*(x[1]-x[0])**2
        return np.array([[t, 8-t], [8-t, t]], dtype=float)


OBJECTIVES = {
    'elongated': Objective('elongated', '实验 A · 学习率与稳定性（图 9.1—9.5）',
        'F(x) = x1² + 25x2²', (0.5, 0.5), (-1, 1, -1, 1),
        np.diag([2., 50.])),
    'rotated': Objective('rotated', '实验 B · 三种算法对比（图 9.4、9.10）',
        'F(x) = 0.5 x^T A x\nA = [[2, 1], [1, 2]]', (0.8, -0.25), (-1, 1, -1, 1),
        np.array([[2., 1.], [1., 2.]])),
    'nonconvex': Objective('nonconvex', '实验 C · 牛顿法与鞍点（图 9.6—9.9）',
        'F(x) = (x2-x1)^4 + 8x1x2 - x1 + x2 + 3',
        (1.5, 0.), (-2, 2, -2, 2)),
}
