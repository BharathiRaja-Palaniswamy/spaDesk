/**
 * @file Draws the quiet parts of a chart: horizontal gridlines with their
 * value labels on the left, and the labels along the bottom.
 */

import { spreadIndexes } from "./scale.js";
import { svg } from "./svg.js";

/** Roughly how wide one character of an axis label is, in pixels, at the label font size. */
const CHARACTER_WIDTH = 6.5;
/** How far below the plot each row of labels sits. */
const TICK_ROW = 16;
const GROUP_ROW = 34;

/**
 * The pixel edges of a chart's plot area.
 * @typedef {{left: number, right: number, top: number, bottom: number}} Frame
 */

/**
 * Adds gridlines and axis labels to a chart.
 *
 * The bottom axis has two rows (see time-axis.js): a short label under each
 * bar, and under those the month or year the bars belong to.
 * @param {SVGElement} root the chart's <svg>
 * @param {object} axes
 * @param {Frame} axes.frame
 * @param {number[]} axes.ticks values to draw a gridline for, lowest first
 * @param {(index: number) => number} axes.x centre of the bar for item `index`
 * @param {(value: number) => number} axes.y vertical position of a value
 * @param {(value: number) => string} axes.formatTick prints a tick value
 * @param {string[]} axes.labels one short label per bar, e.g. "24"
 * @param {{label: string, from: number, to: number}[]} axes.groups runs of
 *   bars that share a month or year, and its name
 */
export function drawAxes(root, { frame, ticks, x, y, formatTick, labels, groups }) {
  for (const tick of ticks) {
    const py = y(tick);
    root.append(
      svg("line", { x1: frame.left, x2: frame.right, y1: py, y2: py, class: tick === 0 ? "baseline" : "grid-line" }),
      label(formatTick(tick), frame.left - 8, py + 4, "end", "axis-label"));
  }

  // Row 1: a label under every bar. Only if the chart is too narrow for that
  // (a phone, or very many bars) are they thinned out, at a regular step.
  const slot = (frame.right - frame.left) / labels.length; // the space each bar has
  const widest = Math.max(...labels.map((text) => text.length)) * CHARACTER_WIDTH + 4;
  const room = Math.max(1, Math.floor((frame.right - frame.left) / widest));
  for (const index of spreadIndexes(labels.length, room)) {
    root.append(label(labels[index], x(index), frame.bottom + TICK_ROW, "middle", "axis-label"));
  }

  // Row 2: the month (or year) each run of bars belongs to, with a short
  // divider where one ends and the next begins.
  for (const group of groups) {
    const left = x(group.from) - slot / 2;
    const right = x(group.to) + slot / 2;
    if (group.from > 0) {
      root.append(svg("line", { x1: left, x2: left, y1: frame.bottom + 4, y2: frame.bottom + GROUP_ROW + 4, class: "axis-divider" }));
    }
    // A run too narrow to hold its name (one bar at the edge of the range) goes without.
    if (right - left >= group.label.length * CHARACTER_WIDTH) {
      root.append(label(group.label, (left + right) / 2, frame.bottom + GROUP_ROW, "middle", "axis-group"));
    }
  }
}

/**
 * One piece of axis text.
 * @param {string} text
 * @param {number} px
 * @param {number} py
 * @param {"start"|"middle"|"end"} anchor which part of the text sits at px
 * @param {string} className "axis-label" or "axis-group" (css/charts.css)
 * @returns {SVGElement}
 */
function label(text, px, py, anchor, className) {
  const node = svg("text", { x: px, y: py, "text-anchor": anchor, class: className });
  node.textContent = text;
  return node;
}
