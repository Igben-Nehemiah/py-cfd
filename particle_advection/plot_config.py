from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class PlotConfig:
    def __init__(self, size: tuple, 
                 title_size: int=12,
                 label_size: int=10,
                 bg_color: str="#aaaaaa",
                 num_colors: int=8,
                 aspect_ratio=1.0):

        self.size = size
        self.title_size = title_size
        self.label_size = label_size
        self.bg_color = bg_color
        self.num_colors = num_colors
        self.aspect_ratio = aspect_ratio