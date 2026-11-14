"""Pure 7-bit ASCII terminal canvas and medical visualization engine.

STRICT ENCODING STANDARD:
All output uses strictly standard 7-bit ASCII characters (ordinal 0-127).
No Unicode box-drawing or special characters (UTF-8 symbols) are permitted,
preventing UnicodeEncodeError crashes on Windows cp1252 consoles and headless CI environments.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


class AsciiCanvas:
    """2D character buffer for pure 7-bit ASCII computer graphics and plotting.

    Parameters
    ----------
    width : int
        Canvas width in characters (columns).
    height : int
        Canvas height in characters (rows).
    fill_char : str
        Background fill character (must be single 7-bit ASCII character, default: ' ').
    """

    def __init__(self, width: int = 80, height: int = 24, fill_char: str = " ") -> None:
        self.width = max(10, int(width))
        self.height = max(5, int(height))
        # Ensure 7-bit ASCII fill
        self.fill_char = fill_char[0] if fill_char and ord(fill_char[0]) < 128 else " "
        self.grid: list[list[str]] = [
            [self.fill_char for _ in range(self.width)] for _ in range(self.height)
        ]

    def set_pixel(self, x: int, y: int, char: str = "*") -> None:
        """Set a single pixel character at (x, y) with coordinate origin at top-left."""
        if 0 <= x < self.width and 0 <= y < self.height:
            ascii_char = char[0] if char and ord(char[0]) < 128 else "*"
            self.grid[y][x] = ascii_char

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, char: str = "*") -> None:
        """Draw line using Bresenham's algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while True:
            self.set_pixel(curr_x, curr_y, char)
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

    def draw_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        border_char: str = "+",
        h_char: str = "-",
        v_char: str = "|",
    ) -> None:
        """Draw rectangular border box in pure 7-bit ASCII."""
        for col in range(x, min(x + w, self.width)):
            self.set_pixel(col, y, h_char)
            self.set_pixel(col, y + h - 1, h_char)
        for row in range(y, min(y + h, self.height)):
            self.set_pixel(x, row, v_char)
            self.set_pixel(x + w - 1, row, v_char)
        # Corners
        self.set_pixel(x, y, border_char)
        self.set_pixel(x + w - 1, y, border_char)
        self.set_pixel(x, y + h - 1, border_char)
        self.set_pixel(x + w - 1, y + h - 1, border_char)

    def draw_text(self, x: int, y: int, text: str) -> None:
        """Draw string horizontally starting at (x, y)."""
        if 0 <= y < self.height:
            for i, ch in enumerate(text):
                col = x + i
                if 0 <= col < self.width:
                    ascii_ch = ch if ord(ch) < 128 else "?"
                    self.grid[y][col] = ascii_ch

    def render(self) -> str:
        """Render canvas grid into newline-delimited ASCII string."""
        return "\n".join("".join(row) for row in self.grid)


def render_sparkline(values: Sequence[float] | np.ndarray, width: int = 40) -> str:
    """Render 1D sequence into single-line 7-bit ASCII density sparkline.

    Uses strictly 7-bit ASCII characters: _ . - = + * # %
    """
    if len(values) == 0:
        return ""
    val_arr = np.asarray(values, dtype=np.float64)
    if len(val_arr) > width:
        indices = np.linspace(0, len(val_arr) - 1, width, dtype=int)
        sampled = val_arr[indices]
    else:
        sampled = val_arr

    min_v = float(np.min(sampled))
    max_v = float(np.max(sampled))
    span = max_v - min_v
    if span < 1e-12:
        return "=" * len(sampled)

    ascii_levels = ["_", ".", "-", "=", "+", "*", "#", "%"]
    chars: list[str] = []
    for v in sampled:
        norm = (v - min_v) / span
        lvl = int(norm * (len(ascii_levels) - 1))
        lvl = max(0, min(lvl, len(ascii_levels) - 1))
        chars.append(ascii_levels[lvl])

    return "".join(chars)


def render_waveform_plot(
    time: np.ndarray,
    signal: np.ndarray,
    title: str = "WAVEFORM",
    width: int = 65,
    height: int = 15,
    y_unit: str = "",
) -> str:
    """Render full 2D ASCII line chart with axes and scale labels."""
    canvas = AsciiCanvas(width=width, height=height)

    # Margins
    left_margin = 10
    bottom_margin = 2
    plot_w = width - left_margin - 3
    plot_h = height - bottom_margin - 2

    # Draw border
    canvas.draw_box(left_margin - 1, 1, plot_w + 2, plot_h + 2, border_char="+", h_char="-", v_char="|")
    canvas.draw_text(left_margin + 2, 0, f"[ {title} ]")

    # Resample signal to plot_w columns
    t_interp = np.linspace(time[0], time[-1], plot_w)
    s_interp = np.interp(t_interp, time, signal)

    min_y = float(np.min(s_interp))
    max_y = float(np.max(s_interp))
    y_span = max_y - min_y
    if y_span < 1e-12:
        y_span = 1.0

    # Draw Y-axis labels
    canvas.draw_text(0, 1, f"{max_y:7.1f}")
    mid_y = 0.5 * (max_y + min_y)
    canvas.draw_text(0, 1 + plot_h // 2, f"{mid_y:7.1f}")
    canvas.draw_text(0, 1 + plot_h, f"{min_y:7.1f}")

    # Plot signal points
    prev_x: int | None = None
    prev_y: int | None = None

    for col in range(plot_w):
        val = s_interp[col]
        # Map val to row inside plot area (row 0 at top, plot_h at bottom)
        norm_y = (val - min_y) / y_span
        row = 1 + int((1.0 - norm_y) * plot_h)
        row = max(1, min(row, 1 + plot_h))
        x_canvas = left_margin + col

        if prev_x is not None and prev_y is not None:
            canvas.draw_line(prev_x, prev_y, x_canvas, row, char="*")
        else:
            canvas.set_pixel(x_canvas, row, char="*")

        prev_x = x_canvas
        prev_y = row

    # Draw X-axis labels
    x_start_label = f"t={time[0]:.2f}s"
    x_end_label = f"t={time[-1]:.2f}s"
    canvas.draw_text(left_margin, height - 1, x_start_label)
    canvas.draw_text(left_margin + plot_w - len(x_end_label), height - 1, x_end_label)

    return canvas.render()


def render_aortic_geometry(
    z: np.ndarray,
    radius: np.ndarray,
    wall_thickness: np.ndarray,
    width: int = 70,
    height: int = 17,
) -> str:
    """Render 2D longitudinal profile of the aorta with aneurysm dilatation in 7-bit ASCII.

    Visualizes upper wall, lumen, centerline, and lower wall.
    """
    canvas = AsciiCanvas(width=width, height=height)

    # Margins
    plot_w = width - 4
    mid_y = height // 2

    # Resample radius and thickness
    z_interp = np.linspace(z[0], z[-1], plot_w)
    r_interp = np.interp(z_interp, z, radius)
    h_interp = np.interp(z_interp, z, wall_thickness)

    max_outer_r = float(np.max(r_interp + h_interp))
    y_scale = (mid_y - 2) / max_outer_r

    canvas.draw_text(2, 0, "+-- INFRARENAL ABDOMINAL AORTA & ANEURYSM PROFILE --+")

    # Draw centerline
    for col in range(plot_w):
        canvas.set_pixel(2 + col, mid_y, char="-")

    for col in range(plot_w):
        r_px = int(r_interp[col] * y_scale)
        h_px = max(1, int(h_interp[col] * y_scale))

        # Upper wall: from (mid_y - r_px - h_px) to (mid_y - r_px)
        upper_outer = mid_y - r_px - h_px
        upper_inner = mid_y - r_px

        # Lower wall: from (mid_y + r_px) to (mid_y + r_px + h_px)
        lower_inner = mid_y + r_px
        lower_outer = mid_y + r_px + h_px

        for y_pos in range(max(1, upper_outer), max(1, upper_inner)):
            canvas.set_pixel(2 + col, y_pos, char="#")
        canvas.set_pixel(2 + col, max(1, upper_inner), char="=")

        for y_pos in range(min(height - 2, lower_inner + 1), min(height - 2, lower_outer + 1)):
            canvas.set_pixel(2 + col, y_pos, char="#")
        canvas.set_pixel(2 + col, min(height - 2, lower_inner), char="=")

        # Lumen interior light shading
        for y_pos in range(upper_inner + 1, lower_inner):
            if y_pos != mid_y:
                canvas.set_pixel(2 + col, y_pos, char=".")

    # Labels
    d_max_mm = float(np.max(radius) * 2000.0)
    d_norm_mm = float(np.min(radius) * 2000.0)
    canvas.draw_text(2, height - 1, f"Healthy D: {d_norm_mm:.1f}mm  |  Max Aneurysm D: {d_max_mm:.1f}mm")

    return canvas.render()


def render_stress_distribution(
    z: np.ndarray,
    stress: np.ndarray,
    strength: np.ndarray,
    width: int = 65,
) -> str:
    """Render spatial wall stress vs wall tensile strength comparison in 7-bit ASCII."""
    num_bins = width - 18
    z_bins = np.linspace(z[0], z[-1], num_bins)
    s_interp = np.interp(z_bins, z, stress) / 1e3     # kPa
    str_interp = np.interp(z_bins, z, strength) / 1e3 # kPa

    lines: list[str] = [
        "+---------------------------------------------------------------+",
        "| SPATIAL WALL STRESS (kPa) vs LOCAL TENSILE STRENGTH (kPa)     |",
        "+---------------------------------------------------------------+",
    ]

    # Print 5 representative sample locations
    step = max(1, len(z_bins) // 5)
    for i in range(0, len(z_bins), step):
        zi_cm = z_bins[i] * 100.0
        si = s_interp[i]
        stri = str_interp[i]
        rpi = si / stri if stri > 0 else 0.0

        bar_len = int(min(30, (si / 600.0) * 30))
        bar = "#" * bar_len + " " * (30 - bar_len)
        flag = "[CRIT]" if rpi >= 0.85 else ("[WARN]" if rpi >= 0.65 else "[ OK ]")
        lines.append(f"| z={zi_cm:4.1f}cm | {bar} | {si:5.1f}/{stri:5.1f} kPa {flag} |")

    lines.append("+---------------------------------------------------------------+")
    return "\n".join(lines)
