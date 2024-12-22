from dataclasses import dataclass

@dataclass
class ColorPalette:
    """
    Eine Sammlung von Farben, die in den Plots verwendet werden können.
    Jede Farbe wird durch ihre RGB-Werte definiert.
    """

    COLORS = {
        "boxplot": (200, 200, 200),  # Hellgrau
        "datapoints_boxplot": (31, 119, 180),  # Blau
        "color_bounds_boxplot": (214, 39, 40),  # Rot
        

        "histogram": (200, 200, 200),  # Hellgrau
        "datapoints_histogram": (31, 119, 180),  # Blau

        "qq_plot": (31, 119, 180),  # Blau
        "degree_45_line": (214, 39, 40),  # Rot
    }

    @staticmethod
    def get_color(name):
        """Gibt die RGB-Werte für eine Farbe zurück."""
        if name in ColorPalette.COLORS:
            return ColorPalette.COLORS[name]
        else:
            raise ValueError(f"Die Farbe '{name}' ist nicht definiert.")

    @staticmethod
    def get_color_hex(name):
        """Gibt die hexadezimale Darstellung einer Farbe zurück."""
        rgb = ColorPalette.get_color(name)
        return "#" + "".join(f"{value:02x}" for value in rgb)