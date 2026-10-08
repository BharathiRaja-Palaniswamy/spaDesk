/**
 * @file How a line chart marks the hovered period: a vertical line, and a dot
 * on each series where it crosses that line.
 */

import { svg } from "./svg.js";

/**
 * Adds a hidden crosshair to a chart and returns the function that moves it.
 * @param {SVGElement} root the chart's <svg>
 * @param {object} chart
 * @param {{top: number, bottom: number}} chart.frame
 * @param {(index: number) => number} chart.x
 * @param {(value: number) => number} chart.y
 * @param {{className: string, values: number[]}[]} chart.series one dot is made per series
 * @returns {(index: ?number) => void} call with a position to show the crosshair
 *   there, or with null to hide it. This is the `mark` that hover.js asks for.
 */
export function createCrosshair(root, { frame, x, y, series }) {
  const line = svg("line", { y1: frame.top, y2: frame.bottom, class: "crosshair" });
  const dots = series.map((s) => svg("circle", { r: 4, class: `dot ${s.className}` }));
  const layer = svg("g", { class: "hover-layer", visibility: "hidden" }, line, ...dots);
  root.append(layer);

  return (index) => {
    if (index === null) {
      layer.setAttribute("visibility", "hidden");
      return;
    }
    const px = x(index);
    line.setAttribute("x1", px);
    line.setAttribute("x2", px);
    dots.forEach((dot, i) => {
      dot.setAttribute("cx", px);
      dot.setAttribute("cy", y(series[i].values[index]));
    });
    layer.setAttribute("visibility", "visible");
  };
}
