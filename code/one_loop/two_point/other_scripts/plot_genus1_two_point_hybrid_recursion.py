#!/usr/bin/env python3
"""Plot the hybrid h/c-recursive genus-one 1->1 amplitude scan as SVG."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
from pathlib import Path


DEFAULT_SCAN_DIR = Path(
    str(package_path('plumbing/results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2'))
)


def _amplitude(t: float, coefficients: tuple[float, float, float]) -> float:
    a, b, c = coefficients
    return (-a * t * t + 2.0 * b * t**4 - c * t**5) / 24.0


def run(scan_dir: Path) -> Path:
    fit = json.loads(
        (scan_dir / "bry_postfreeze_fit.json").read_text(encoding="utf-8")
    )
    rows = fit["rows"]
    fitted = fit["weighted_fit"]["coefficients"]
    coefficients = (float(fitted["a"]), float(fitted["b"]), float(fitted["c"]))

    width, height = 820.0, 650.0
    left, right = 96.0, 28.0
    top_main, bottom_main = 58.0, 405.0
    top_ratio, bottom_ratio = 452.0, 585.0
    y_min, y_max = -0.0082, 0.0002
    ratio_min, ratio_max = 0.90, 1.10

    def map_x(value: float) -> float:
        return left + value * (width - left - right)

    def map_y(value: float) -> float:
        return top_main + (y_max - value) / (y_max - y_min) * (bottom_main - top_main)

    def map_ratio(value: float) -> float:
        return top_ratio + (ratio_max - value) / (ratio_max - ratio_min) * (
            bottom_ratio - top_ratio
        )

    plot_t = [index / 800.0 for index in range(1, 800)]
    analytic = [_amplitude(value, (1.0, 1.0, 1.0)) for value in plot_t]
    fitted_curve = [_amplitude(value, coefficients) for value in plot_t]

    def polyline(values: list[float], mapper) -> str:
        return " ".join(
            f"{map_x(x):.2f},{mapper(y):.2f}" for x, y in zip(plot_t, values)
        )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#27323a}.grid{stroke:#d8dde2;stroke-width:.7}.axis{stroke:#4f5961;stroke-width:1}.tick{font-size:13px}.label{font-size:16px}.legend{font-size:13px}</style>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="18">Genus-one 1→1: necklace h-recursion + OPE c-recursion</text>',
    ]

    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x_pixel = map_x(tick)
        svg.append(
            f'<line class="grid" x1="{x_pixel:.2f}" y1="{top_main}" x2="{x_pixel:.2f}" y2="{bottom_ratio}"/>'
        )
        svg.append(
            f'<text class="tick" x="{x_pixel:.2f}" y="{bottom_ratio+24:.2f}" text-anchor="middle">{tick:.1f}</text>'
        )
    for tick in (-0.008, -0.006, -0.004, -0.002, 0.0):
        y_pixel = map_y(tick)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y_pixel:.2f}" x2="{width-right}" y2="{y_pixel:.2f}"/>'
        )
        svg.append(
            f'<text class="tick" x="{left-10:.2f}" y="{y_pixel+4:.2f}" text-anchor="end">{tick:.3f}</text>'
        )
    for tick in (0.90, 0.95, 1.00, 1.05, 1.10):
        y_pixel = map_ratio(tick)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y_pixel:.2f}" x2="{width-right}" y2="{y_pixel:.2f}"/>'
        )
        svg.append(
            f'<text class="tick" x="{left-10:.2f}" y="{y_pixel+4:.2f}" text-anchor="end">{tick:.2f}</text>'
        )

    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top_main}" x2="{left}" y2="{bottom_main}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom_main}" x2="{width-right}" y2="{bottom_main}"/>',
            f'<line class="axis" x1="{left}" y1="{top_ratio}" x2="{left}" y2="{bottom_ratio}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom_ratio}" x2="{width-right}" y2="{bottom_ratio}"/>',
            f'<polyline points="{polyline(analytic, map_y)}" fill="none" stroke="#315b9d" stroke-width="2.2" stroke-dasharray="8 5"/>',
            f'<polyline points="{polyline(fitted_curve, map_y)}" fill="none" stroke="#d36c45" stroke-width="2.1"/>',
            f'<line x1="{left}" y1="{map_ratio(1.0):.2f}" x2="{width-right}" y2="{map_ratio(1.0):.2f}" stroke="#315b9d" stroke-width="1.5" stroke-dasharray="7 5"/>',
        ]
    )

    for row in rows:
        t = float(row["t"])
        value = float(row["bry_minus_i_amplitude"])
        error = float(row["bry_rqmc_standard_error"])
        analytic_value = float(row["analytic_bry_minus_i_amplitude"])
        ratio = value / analytic_value
        ratio_error = error / abs(analytic_value)
        x_pixel = map_x(t)
        y_pixel = map_y(value)
        y_low, y_high = map_y(value - error), map_y(value + error)
        ratio_pixel = map_ratio(ratio)
        ratio_low = map_ratio(ratio - ratio_error)
        ratio_high = map_ratio(ratio + ratio_error)
        svg.extend(
            [
                f'<line x1="{x_pixel:.2f}" y1="{y_low:.2f}" x2="{x_pixel:.2f}" y2="{y_high:.2f}" stroke="#9e2f28" stroke-width="1.5"/>',
                f'<circle cx="{x_pixel:.2f}" cy="{y_pixel:.2f}" r="4.4" fill="#9e2f28"/>',
                f'<line x1="{x_pixel:.2f}" y1="{ratio_low:.2f}" x2="{x_pixel:.2f}" y2="{ratio_high:.2f}" stroke="#9e2f28" stroke-width="1.5"/>',
                f'<circle cx="{x_pixel:.2f}" cy="{ratio_pixel:.2f}" r="4.0" fill="#9e2f28"/>',
            ]
        )

    legend_x = width - 260.0
    svg.extend(
        [
            f'<line x1="{legend_x}" y1="78" x2="{legend_x+42}" y2="78" stroke="#315b9d" stroke-width="2.2" stroke-dasharray="8 5"/><text class="legend" x="{legend_x+51}" y="83">analytic amplitude</text>',
            f'<line x1="{legend_x}" y1="101" x2="{legend_x+42}" y2="101" stroke="#d36c45" stroke-width="2.1"/><text class="legend" x="{legend_x+51}" y="106">hybrid-recursion fit</text>',
            f'<circle cx="{legend_x+21}" cy="124" r="4.4" fill="#9e2f28"/><text class="legend" x="{legend_x+51}" y="129">worldsheet RQMC</text>',
            f'<text x="{width-right-55}" y="157" text-anchor="end" font-size="13">(a,b,c)=({coefficients[0]:.4f}, {coefficients[1]:.4f}, {coefficients[2]:.4f})</text>',
            f'<text class="label" x="{width/2:.1f}" y="{height-18}" text-anchor="middle">t = −iω</text>',
            f'<text class="label" x="24" y="{(top_main+bottom_main)/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {(top_main+bottom_main)/2:.1f})">−i A⁽¹⁾₁→₁(it)</text>',
            f'<text x="30" y="{(top_ratio+bottom_ratio)/2-8:.1f}" text-anchor="middle" font-size="13" transform="rotate(-90 30 {(top_ratio+bottom_ratio)/2-8:.1f})">worldsheet / analytic</text>',
            '</svg>',
        ]
    )

    output = scan_dir / "genus1_two_point_hybrid_recursion.svg"
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    return out


if __name__ == "__main__":
    output = run(parser().parse_args().scan_dir)
    print(f"wrote {output}")
