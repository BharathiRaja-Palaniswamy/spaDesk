/**
 * @file Draws a column chart as SVG: one vertical bar per period, its height
 * showing that period's value. Used for values over time, such as revenue
 * per day.
 *
 * The chart is drawn at the pixel width of its container, so text is never
 * stretched. main.js redraws it when the window is resized.
 */

import { el } from "../dom.js";
import { drawAxes } from "./axes.js";
import { attachHover } from "./hover.js";
import { axisTicks, linear } from "./scale.js";
import { svg } from "./svg.js";

const HEIGHT = 260;
/** Space around the plot for the axis labels. The bottom holds two rows of them. */
const MARGIN = { top: 12, right: 16, bottom: 48, left: 52 };
/** Bars never get wider than this, however few there are; the rest of the slot is left as air. */
const MAX_BAR_WIDTH = 24;
/** How round the top corners of a bar are. The bottom stays square, on the baseline. */
const CORNER = 4;

/**
 * The outline of one bar: square at the bottom, rounded at the top.
 * @param {number} left the bar's left edge
 * @param {number} top the bar's top edge
 * @param {number} width
 * @param {number} bottom the baseline
 * @returns {string} an SVG path
 */
function barPath(left, top, width, bottom) {
  const right = left + width;
  const r = Math.min(CORNER, width / 2, bottom - top); // a very short bar gets a smaller corner
  // "Q" draws a curve: it bends round the corner point to the next point.
  return `M${left},${bottom} L${left},${top + r} Q${left},${top} ${left + r},${top} `
    + `L${right - r},${top} Q${right},${top} ${right},${top + r} L${right},${bottom} Z`;
}

/**
 * Draws a column chart into `body`, replacing whatever was there.
 * @param {HTMLElement} body the card body to draw into
 * @param {object} chart
 * @param {string[]} chart.labels the full name of each bar's period, for the tooltip, e.g. "27 Sept"
 * @param {{ticks: string[], groups: object[]}} chart.axis the labels along the bottom, from timeAxis()
 * @param {number[]} chart.values one per label; the bar heights
 * @param {string} chart.name what the values are, shown in the tooltip, e.g. "Revenue"
 * @param {string} chart.className picks the bar colour (css/charts.css), e.g. "series-1"
 * @param {(value: number) => string} chart.formatValue prints a value in the tooltip
 * @param {(value: number) => string} chart.formatTick prints a value on the left axis
 */
export function drawColumnChart(body, { labels, axis, values, name, className, formatValue, formatTick }) {
  const width = Math.max(body.clientWidth, 320);
  const frame = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: HEIGHT - MARGIN.bottom };
  const slot = (frame.right - frame.left) / labels.length; // the space each bar has
  const barWidth = Math.min(MAX_BAR_WIDTH, slot * 0.72);
  const x = (index) => frame.left + slot * (index + 0.5);  // centre of bar `index`
  const ticks = axisTicks(Math.max(...values));
  const y = linear(0, ticks.at(-1), frame.bottom, frame.top); // value -> pixels

  // tabindex makes the chart reachable by keyboard; hover.js handles the arrow keys.
  const root = svg("svg", { width, height: HEIGHT, class: "chart", tabindex: 0, role: "img",
    "aria-label": `${name} per period. Use the table view for the numbers.` });
  drawAxes(root, { frame, ticks, x, y, formatTick, labels: axis.ticks, groups: axis.groups });

  // A soft band behind the hovered bar.
  const highlight = svg("rect", { y: frame.top, width: slot, height: frame.bottom - frame.top,
    class: "column-highlight", visibility: "hidden" });
  root.append(highlight);

  values.forEach((value, index) => {
    if (value <= 0) return; // nothing to draw for an empty period
    root.append(svg("path", { d: barPath(x(index) - barWidth / 2, y(value), barWidth, frame.bottom),
      class: `bar ${className}` }));
  });

  // The tooltip is positioned inside this box.
  const plot = el("div", { className: "plot" }, root);
  body.replaceChildren(plot);
  attachHover(root, plot, {
    count: labels.length, frame, x,
    mark: (index) => {
      highlight.setAttribute("visibility", index === null ? "hidden" : "visible");
      if (index !== null) highlight.setAttribute("x", x(index) - slot / 2);
    },
    content: (index) => ({
      title: labels[index],
      rows: [{ className, value: formatValue(values[index]), name }],
    }),
  });
}
