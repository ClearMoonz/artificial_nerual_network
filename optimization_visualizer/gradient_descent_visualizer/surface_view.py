"""3D objective surface and trajectories from the existing iteration history."""
import numpy as np
from optimizers import NAMES

COLORS = {'sd': '#2563eb', 'newton': '#e67722', 'cg': '#15977e'}
DEFAULT_CAMERA = (28., -62., 0.)


class SurfacePlot:
    def __init__(self, axes, objective, bounds, runs, focus=True, camera=DEFAULT_CAMERA):
        self.axes, self.objective, self.bounds = axes, objective, tuple(bounds)
        self.focus = focus
        xmin, xmax, ymin, ymax = bounds
        self.X, self.Y = np.meshgrid(np.linspace(xmin, xmax, 45), np.linspace(ymin, ymax, 45))
        self.Z = objective.grid(self.X, self.Y)
        low, high = float(self.Z.min()), float(self.Z.max())
        padding = max((high-low)*.025, .05)
        self.clipped = focus and not objective.quadratic and high > 12.
        if self.clipped:
            high = 12.
            padding = max((high-low)*.04, .1)
        self.z_range = (low-padding, high+padding)
        # Hide the high portion rather than flattening it into a false plateau.
        visible_z = np.where(self.Z <= high, self.Z, np.nan) if self.clipped else self.Z
        axes.plot_surface(self.X, self.Y, visible_z, cmap='Blues', alpha=.42,
                          linewidth=0, antialiased=True, rcount=45, ccount=45,
                          vmin=low, vmax=high, zorder=1)
        levels = np.linspace(low, high, 9)
        axes.contour(self.X, self.Y, self.Z, levels=levels,
                     zdir='z', offset=self.z_range[0], colors='#8296a8',
                     linewidths=.55, alpha=.65, zorder=0)
        axes.set(xlim=(xmin,xmax), ylim=(ymin,ymax), zlim=self.z_range,
                 xlabel='x1', ylabel='x2', zlabel='F(x1, x2)',
                 title='三维曲面与迭代轨迹 · 拖动旋转')
        axes.set_box_aspect((1, 1, .72))
        axes.computed_zorder = False
        axes.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        axes.mouse_init(rotate_btn=1, pan_btn=2, zoom_btn=3)
        axes.tick_params(labelsize=8)
        axes.xaxis.pane.set_facecolor('#f7f9fc')
        axes.yaxis.pane.set_facecolor('#f7f9fc')
        axes.zaxis.pane.set_facecolor('#f7f9fc')
        self.lines, self.current = {}, {}
        for run in runs:
            self.lines[run.method], = axes.plot([],[],[],color=COLORS[run.method],
                lw=2.3,marker='o',markersize=3,label=NAMES[run.method],zorder=4)
            self.current[run.method], = axes.plot([],[],[],color=COLORS[run.method],
                marker='o',markersize=7,markeredgecolor='white',markeredgewidth=.8,zorder=5)
        start = runs[0].history[0]['x'] if runs else [0.,0.]
        axes.plot([start[0]],[start[1]],[objective.value(start)],marker='*',
                  color='#243449',markersize=10,label='起点',zorder=6)
        axes.legend(loc='upper left', fontsize=8, framealpha=.88)
        self.note = axes.text2D(.02, .01, '', transform=axes.transAxes,
                               fontsize=8, color='#52677a')
        self.update(runs)

    @property
    def camera(self):
        return (self.axes.elev, self.axes.azim, self.axes.roll)

    def reset_camera(self):
        self.axes.view_init(elev=DEFAULT_CAMERA[0], azim=DEFAULT_CAMERA[1], roll=DEFAULT_CAMERA[2])

    def update(self, runs):
        xmin,xmax,ymin,ymax = self.bounds
        lo,hi = self.z_range
        outside = []
        for run in runs:
            xy = np.asarray([state['x'] for state in run.history])
            # Use the recorded, unrounded function values, not a fitted mesh height.
            z = np.asarray([state['f'] for state in run.history])
            visible = ((xy[:,0]>=xmin)&(xy[:,0]<=xmax)&(xy[:,1]>=ymin)&
                       (xy[:,1]<=ymax)&(z>=lo)&(z<=hi))
            data = np.column_stack((xy,z))
            plotted = np.where(visible[:,None], data, np.nan)
            self.lines[run.method].set_data_3d(*plotted.T)
            self.current[run.method].set_data_3d(*plotted[-1:, :].T)
            if not visible[-1]:
                outside.append(NAMES[run.method])
        notes = []
        if self.clipped:
            notes.append('聚焦谷底：F > 12 的高处隐藏；数据不裁切')
        if outside:
            notes.append('当前点超出图域：'+'、'.join(outside))
        self.note.set_text('\n'.join(notes))
