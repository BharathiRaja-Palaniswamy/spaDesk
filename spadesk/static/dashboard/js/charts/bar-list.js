/**
 * @file Draws a horizontal bar list: one row per item, with its name on the
 * left, a bar whose length shows its value, and the value printed at the end.
 *
 * It suits named things being compared by size (categories, services), where
 * the names are too long to sit under vertical bars. Bars normally share one
 * colour, since length already shows the value. A row is given its own
 * colour only when the items are things with an identity of their own (the
 * channels, for example), which each keep a fixed colour.
 */

import { el } from "../dom.js";

/**
 * Draws the list into `body`, replacing whatever was there.
 * @param {HTMLElement} body the card body to draw into
 * @param {object} chart
 * @param {{label: string, value: number, text: string, detail: string, className?: string}[]} chart.rows
 *   in display order. `value` sets the bar length, `text` is printed at the
 *   end of the bar, and `detail` appears as a tooltip on hover or keyboard
 *   focus. A row may name its own colour class; otherwise it uses `chart.className`.
 * @param {string} [chart.className] the bar colour for every row (css/charts.css), e.g. "series-1"
 */
export function drawBarList(body, { rows, className }) {
  const largest = Math.max(...rows.map((row) => row.value), 1); // 1 avoids dividing by zero
  body.replaceChildren(el("div", { className: "bar-list" }, ...rows.map((row) => {
    const bar = el("span", { className: `bar-list-bar ${row.className ?? className}` });
    // The bar's length is the one thing set from script, because it comes from
    // the data. Everything about how a bar looks is in css/bar-list.css.
    bar.style.width = `${(row.value / largest) * 100}%`;

    const line = el("div", { className: "bar-list-row has-tooltip", tabIndex: 0 },
      el("span", { className: "bar-list-label", textContent: row.label }),
      el("span", { className: "bar-list-track" }, bar),
      el("span", { className: "bar-list-value", textContent: row.text }));
    line.dataset.tooltip = row.detail; // shown by css/tooltip.css
    return line;
  })));
}
