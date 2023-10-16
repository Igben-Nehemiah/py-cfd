from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class PlotConfig:
    def __init__(self, size: tuple, 
                 title_size: int=8,
                 axes_title_size: int=7,
                 label_size: int=7,
                 bg_color: str="#000",
                 num_colors: int=8,
                 aspect_ratio=1.0):

        self.size = size
        self.title_size = title_size
        self.axes_title_size = axes_title_size
        self.label_size = label_size
        self.bg_color = bg_color
        self.num_colors = num_colors
        self.aspect_ratio = aspect_ratio