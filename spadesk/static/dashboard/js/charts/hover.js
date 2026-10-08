/**
 * @file The hover behaviour of a chart that has one position per period (the
 * line chart): find the period nearest the pointer, show a tooltip for it,
 * and let the chart mark it in its own way.
 *
 * The reader aims at a period, not at a thin mark: the pointer only has to be
 * nearest to it, anywhere over the chart. Left/right arrow keys do the same
 * for keyboard users.
 */

import { el } from "../dom.js";

/**
 * Adds hover and keyboard behaviour to a chart.
 * @param {SVGElement} root the chart's <svg>; it listens for the pointer and keys
 * @param {HTMLElement} plot the box around the chart; the tooltip is added to it
 * @param {object} chart
 * @param {number} chart.count how many positions there are
 * @param {{left: number, right: number}} chart.frame plot edges, used to keep the tooltip inside
 * @param {(index: number) => number} chart.x pixel position of position `index`
 * @param {(index: ?number) => void} chart.mark highlights that position in the
 *   chart's own way; called with null to clear it
 * @param {(index: number) => {title: string, rows: {className: string, value: string, name: string}[]}} chart.content
 *   what the tooltip shows for a position
 */
export function attachHover(root, plot, { count, frame, x, mark, content }) {
  const tooltip = el("div", { className: "chart-tooltip", hidden: true });
  plot.append(tooltip);
  /** The position currently shown, or null. Arrow keys move from here. */
  let current = null;

  /** Marks a position and shows its tooltip. */
  function show(index) {
    current = index;
    mark(index);
    const { title, rows } = content(index);
    tooltip.replaceChildren(
      el("div", { className: "chart-tooltip-title", textContent: title }),
      // The value is bold and comes first; the series name follows.
      ...rows.map((row) => el("div", { className: "chart-tooltip-row" },
        el("span", { className: `key ${row.className}` }),
        el("strong", { textContent: row.value }),
        el("span", { textContent: row.name }))));
    tooltip.hidden = false;
    // Position is the one thing that has to be set from script, because it
    // follows the pointer. All of the tooltip's looks are in css/charts.css.
    const px = x(index);
    tooltip.style.left = `${px}px`;
    tooltip.classList.toggle("left-of-pointer", px > (frame.left + frame.right) / 2); // keep it inside the chart
  }

  /** Clears the mark and hides the tooltip. */
  function hide() {
    current = null;
    mark(null);
    tooltip.hidden = true;
  }

  /**
   * Which position is closest to the pointer.
   * @param {PointerEvent} event
   * @returns {number}
   */
  function nearest(event) {
    const px = event.clientX - root.getBoundingClientRect().left;
    const step = count > 1 ? x(1) - x(0) : 1;
    return Math.min(count - 1, Math.max(0, Math.round((px - x(0)) / step)));
  }

  root.addEventListener("pointermove", (event) => show(nearest(event)));
  root.addEventListener("pointerleave", hide);
  root.addEventListener("blur", hide);
  root.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const move = event.key === "ArrowRight" ? 1 : -1;
    show(Math.min(count - 1, Math.max(0, (current ?? (move > 0 ? -1 : count)) + move)));
  });
}
