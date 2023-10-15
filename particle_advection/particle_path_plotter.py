import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from vector_field import VectorField, GridRange
import numpy as np

from particle_advection.trace_method import TraceMethod
from particle_advection.plot_config import PlotConfig
from particle_advection.particle_tracer import ParticleTracer

class Plotter:
    def __init__(self, plot_config: PlotConfig, particle_tracer: ParticleTracer):
        self.cfg = plot_config
        self.figure = plt.figure(figsize=self.cfg.size)
        self.p_tracer = particle_tracer
    
    def get_axes(self):
        gs = GridSpec(1,1)
        return self.figure.add_subplot(gs[0,0])
    
    def update_axes(self, axes):
        title = "Particle Tracing In Vector Field"
        axes.set_title(title, fontsize=self.cfg.title_size)

        axes.set_xlabel('$x$')
        axes.set_ylabel('$y$')
        axes.set_aspect('equal')

        return axes
    
    def make_background(self, axes):
        grids_ranges = []

        for grid_range in self.p_tracer.field.grid_ranges:
            grids_ranges.append(grid_range.start)
            grids_ranges.append(grid_range.end)


        return axes.imshow(np.sum(self.p_tracer.field.mesh_grids_values, axis=0), 
                           extent=grids_ranges)
    
    def make_quiver(self, axes):
        return axes.quiver(*self.p_tracer.field.mesh_grids,
                            *self.p_tracer.field.mesh_grids_values, 
                            np.sum(self.p_tracer.field.mesh_grids_values, axis=0))
    
    def plot(self, animate=False, show_paths=False):
        if (self.p_tracer.field.dim != 2):
            raise ValueError('This can only work for 2D fields for now!')
        
        if animate:
            self.__plot_animated(show_paths)
        else:
            self.__plot_static()

    def __plot_animated(self, show_paths=False):
        scatter_cfg = {
            "color": ['y'], 
            "marker": ".",
        }
        xlim = self.p_tracer.field.grid_ranges[0].start, self.p_tracer.field.grid_ranges[0].end
        ylim = self.p_tracer.field.grid_ranges[1].start, self.p_tracer.field.grid_ranges[1].end

        axes = self.update_axes(self.get_axes())
        self.make_background(axes)
        self.make_quiver(axes)

        self.p_tracer.trace(method=TraceMethod.RK4)
        positions_evolution = self.p_tracer.positions_evolution

        paths = axes.scatter(x=positions_evolution[0,:,0], y=positions_evolution[0,:,1], **scatter_cfg)

        def update(frame):
            axes.clear()
            axes.set_xlim(*xlim)
            axes.set_ylim(*ylim)
            axes.set_facecolor("k")
            # self.make_background(axes)
            self.make_quiver(axes)

            axes.scatter(x=positions_evolution[frame,:,0], y=positions_evolution[frame,:,1], **scatter_cfg)

            if show_paths:
                lines = axes.plot(positions_evolution[:frame,:,0], positions_evolution[:frame,:,1], color='b')
                return paths, lines
            return paths,


        anim = FuncAnimation(self.figure, update, len(positions_evolution),
                            interval=1, cache_frame_data=False)
        plt.show()
        

    def __plot_static(self):
        axes = self.update_axes(self.get_axes())

        self.make_background(axes)
        self.make_quiver(axes)
        plt.show()



def run():
    # Create vector field
    field = VectorField([GridRange(-5, 5, 30), GridRange(-5, 5, 30)])
    # field.set_from_function(lambda x, y: (x**2, y**2))
    # field.set_from_function(lambda x, y: (x, y))
    # field.set_from_function(lambda x, y: (np.sin(x + y), np.cos(x + y)))
    # field.set_from_function(lambda x, y: (-y, x))
    field.set_from_function(lambda x, y: (2*x*y, 1-x**2-y**2))

    # Initialise ParticleTracer with created vector field
    initial_positions = [(0, 0.5), (0, 0.6), (0, 0.7), (0, 0.8), (0, 0.9), 
                         (-1, -0.2), (-1, -0.1), (-1, 0), (-1, 0.1), (-1, 0.2),
                         (-2,4)]
    tracer = ParticleTracer(field=field, initial_positions=initial_positions, dt=0.05, n_time_steps=200)

    plot_cfg = PlotConfig(size=(10,8),
        title_size=20,
        label_size=16)
    
    plotter = Plotter(plot_cfg, tracer)
    plotter.plot(animate=True, show_paths=True)

run()