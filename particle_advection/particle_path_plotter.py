import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.quiver import Quiver
from vector_field import VectorField
from itertools import cycle
from typing import Sequence

from particle_advection.trace_method import TraceMethod
from particle_advection.plot_config import PlotConfig
from particle_advection.particle_tracer import ParticleTracer

class Plotter:
    def __init__(self, plot_config: PlotConfig, particle_tracers: Sequence[ParticleTracer]):
        """
        Initialise the Plotter.

        Parameters
        ----------
        - plot_config (PlotConfig): The configuration for plotting.
        - particle_tracers (Sequence[ParticleTracer]): A list of ParticleTracer objects.

        """
        self.cfg = plot_config
        self.figure = plt.figure(figsize=self.cfg.size, facecolor=self.cfg.bg_color)
        self.p_tracers = particle_tracers
        self.__set_axes_array(2, 2)

    def __set_axes_array(self, n_rows: int, n_cols: int):
        """
        Set up an array of axes for subplots.

        Parameters
        ----------
        - n_rows (int): Number of rows in the subplot grid.
        - n_cols (int): Number of columns in the subplot grid.

        """
        self.__axes_list: list[Axes] = []
        gs = GridSpec(n_rows,n_cols)

        for row in range(n_rows):
            for col in range(n_cols):
                self.__axes_list.append(self.figure.add_subplot(gs[row, col]))
    
    def update_axes(self, axes: Axes, title=""):
        """
        Update the properties of an axes.

        Parameters
        ----------
        - axes (Axes): The axes to be updated.
        - title (str): Title for the axes.

        Returns
        -------
        Axes: The updated axes.

        """
        axes.set_title(title, fontsize=self.cfg.axes_title_size, color="#fff")
        axes.set_aspect(self.cfg.aspect_ratio)
    
        return axes
    
    def make_quiver(self, field: VectorField, axes: Axes) -> Quiver:
        """
        Create a quiver plot on the given axes.

        Parameters
        ----------
        - field (VectorField): The vector field.
        - axes (Axes): The axes for the quiver plot.

        Returns
        -------
        Quiver: The quiver plot.

        """
        return axes.quiver(*field.mesh_grids,
                            *field.mesh_grids_values, 
                            np.sum(field.mesh_grids_values, axis=0))
    
    def plot(self, animate=False, show_path_lines: Sequence[bool]=[False], save=False) -> None:
        """
        Plot vector fields.

        Parameters
        ----------
        - animate (bool): If True, create an animated plot.
        - show_path_lines (Sequence[bool]): List of booleans to control path lines for each vector field.
        - save (bool): If True, save the plot.

        """
        if len(show_path_lines) == 0: show_path_lines = [False]
        
        for p_tracer in self.p_tracers:
            if (p_tracer.field.dim != 2):
                raise ValueError('This can only work for 2D fields for now!')

        if animate:
            self.__plot_animated(show_path_lines, save)
        else:
            self.__plot_static(save)

    def __plot_animated(self, show_path_lines: Sequence[bool], save=False):
        """
        Create an animated plot.

        Parameters
        ----------
        - show_path_lines (Sequence[bool]): List of booleans to control path lines for each vector field.
        - save (bool): If True, save the animation.
        """
        scatter_cfg = {
            "color": 'y', 
            "marker": ".",
        }

        xlims = []
        ylims = []
        positions_evolutions = []
        self.figure.suptitle('Vector Fields', fontsize=self.cfg.title_size, color='#fff')

        for pt in self.p_tracers:
            xlims.append(pt.field.grid_ranges[0].range())
            ylims.append(pt.field.grid_ranges[1].range())

            pt.trace(method=TraceMethod.RK4)
            positions_evolutions.append(pt.positions_evolution)
        
        def update(frame: int) -> None:
            for i, (ax, show_path) in enumerate(zip(self.__axes_list, cycle(show_path_lines))):
                ax.clear()
                ax = self.update_axes(ax, self.p_tracers[i].field.desc)
                ax.set_xlim(*xlims[i])
                ax.set_ylim(*ylims[i])
                ax.set_facecolor("k")
                self.make_quiver(self.p_tracers[i].field, ax)

                ax.scatter(x=positions_evolutions[i][frame,:,0], y=positions_evolutions[i][frame,:,1], **scatter_cfg)

                if show_path:
                    ax.plot(positions_evolutions[i][:frame,:,0], 
                            positions_evolutions[i][:frame,:,1], color='b', alpha=0.85, linestyle="--")
                  
                  

        writer = FFMpegWriter(fps=5, bitrate=-1)
        anim = FuncAnimation(self.figure, update, len(positions_evolutions[0]),
                                interval=1, cache_frame_data=False)
        if save:
            anim.save('particle_advection.mp4', writer)

        plt.show()
        

    def __plot_static(self, save=False):
        """
        Create an static plot of vector field(s).

        Parameters
        ----------
        - save (bool): If True, save plot as png.

        """
        for i, pt in enumerate(self.p_tracers):
            axes = self.update_axes(self.__axes_list[i])
            self.make_quiver(pt.field, axes)
        
        if save:
            self.figure.savefig('fields.png')
        plt.show()