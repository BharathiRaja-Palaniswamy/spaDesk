/**
 * @file Draws the quiet parts of a chart: horizontal gridlines with their
 * value labels on the left, and the labels along the bottom.
 */

import { spreadIndexes } from "./scale.js";
import { svg } from "./svg.js";

/** At most this many labels are printed along the bottom, so they never overlap. */
const MAX_X_LABELS = 6;

/**
 * The pixel edges of a chart's plot area.
 * @typedef {{left: number, right: number, top: number, bottom: number}} Frame
 */

/**
 * Adds gridlines and axis labels to a chart.
 * @param {SVGElement} root the chart's <svg>
 * @param {object} axes
 * @param {Frame} axes.frame
 * @param {number[]} axes.ticks values to draw a gridline for, lowest first
 * @param {(index: number) => number} axes.x position of the label for item `index`
 * @param {(value: number) => number} axes.y vertical position of a value
 * @param {string[]} axes.labels one per item, along the bottom
 * @param {(value: number) => string} axes.formatTick prints a tick value
 */
export function drawAxes(root, { frame, ticks, x, y, labels, formatTick }) {
  for (const tick of ticks) {
    const py = y(tick);
    root.append(
      svg("line", { x1: frame.left, x2: frame.right, y1: py, y2: py, class: tick === 0 ? "baseline" : "grid-line" }),
      label(formatTick(tick), frame.left - 8, py + 4, "end"));
  }
  const last = labels.length - 1;
  for (const index of spreadIndexes(labels.length, MAX_X_LABELS)) {
    // The line starts and ends at the plot's edges, so the first and last
    // labels are pulled inward to stay inside the chart.
    const anchor = index === 0 ? "start" : index === last && last > 0 ? "end" : "middle";
    root.append(label(labels[index], x(index), frame.bottom + 18, anchor));
  }
}

/**
 * One axis label.
 * @param {string} text
 * @param {number} px
 * @param {number} py
 * @param {"start"|"middle"|"end"} anchor which part of the text sits at px
 * @returns {SVGElement}
 */
function label(text, px, py, anchor) {
  const node = svg("text", { x: px, y: py, "text-anchor": anchor, class: "axis-label" });
  node.textContent = text;
  return node;
}
