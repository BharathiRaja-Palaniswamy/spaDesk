/**
 * @file Draws a line chart as SVG.
 *
 * The chart is drawn at the pixel width of its container, so text is never
 * stretched. main.js redraws it when the window is resized.
 */

import { el } from "../dom.js";
import { drawAxes } from "./axes.js";
import { createCrosshair } from "./crosshair.js";
import { attachHover } from "./hover.js";
import { axisTicks, linear } from "./scale.js";
import { svg } from "./svg.js";

const HEIGHT = 240;
/** Space around the plot for the axis labels. */
const MARGIN = { top: 12, right: 16, bottom: 30, left: 52 };

/**
 * Draws a line chart into `body`, replacing whatever was there.
 * @param {HTMLElement} body the card body to draw into
 * @param {object} chart
 * @param {string[]} chart.labels one per point, along the bottom, e.g. ["1 Sep", "2 Sep"]
 * @param {{name: string, className: string, values: number[]}[]} chart.series one line
 *   each; `className` picks the colour (css/charts.css); `values` has one number per label
 * @param {(value: number) => string} chart.formatValue prints a value in the tooltip
 * @param {(value: number) => string} chart.formatTick prints a value on the left axis
 * @param {boolean} [chart.area] soft fill under the line (use with a single series)
 */
export function drawLineChart(body, { labels, series, formatValue, formatTick, area = false }) {
  const width = Math.max(body.clientWidth, 320);
  const frame = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: HEIGHT - MARGIN.bottom };

  const ticks = axisTicks(Math.max(...series.flatMap((s) => s.values)));
  const x = linear(0, Math.max(labels.length - 1, 1), frame.left, frame.right); // point index -> pixels
  const y = linear(0, ticks.at(-1), frame.bottom, frame.top);                   // value -> pixels

  // tabindex makes the chart reachable by keyboard; hover.js handles the arrow keys.
  const root = svg("svg", { width, height: HEIGHT, class: "chart", tabindex: 0, role: "img",
    "aria-label": `${series.map((s) => s.name).join(", ")} over time. Use the table view for the numbers.` });

  drawAxes(root, { frame, ticks, x, y, labels, formatTick });

  for (const s of series) {
    const points = s.values.map((value, index) => `${x(index)},${y(value)}`);
    if (area) {
      // The same outline as the line, closed down to the baseline.
      const outline = `M${frame.left},${frame.bottom} L${points.join(" L")} L${x(labels.length - 1)},${frame.bottom} Z`;
      root.append(svg("path", { d: outline, class: `area ${s.className}` }));
    }
    // In an SVG path, "M" moves to the first point and "L" draws a line to the next.
    root.append(svg("path", { d: `M${points.join(" L")}`, class: `line ${s.className}` }));
  }

  // The tooltip is positioned inside this box.
  const plot = el("div", { className: "plot" }, root);
  body.replaceChildren(plot);
  attachHover(root, plot, {
    count: labels.length, frame, x,
    mark: createCrosshair(root, { frame, x, y, series }),
    content: (index) => ({
      title: labels[index],
      rows: series.map((s) => ({ className: s.className, value: formatValue(s.values[index]), name: s.name })),
    }),
  });
}
