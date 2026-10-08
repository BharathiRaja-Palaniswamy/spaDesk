/**
 * @file A chart card: title, a one-line explanation, the chart, and a button
 * that swaps the chart for a table of the same numbers.
 */

import { el } from "./dom.js";
import { dataTable } from "./table.js";

/**
 * Remembers, per card title, whether the visitor switched to the table, so
 * the choice survives a filter change (which redraws every card).
 * @type {Map<string, boolean>}
 */
const showingTable = new Map();

/**
 * Builds a card inside `container`, replacing whatever was there.
 *
 * @param {Element} container the card's box from index.html
 * @param {object} card
 * @param {string} card.title
 * @param {string} card.subtitle one line saying what the chart shows
 * @param {boolean} card.empty true shows "nothing to show" instead of the chart
 * @param {(body: HTMLElement) => void} card.draw draws the chart into `body`
 * @param {{columns: string[], rows: string[][]}} card.table the same data as a table
 */
export function chartCard(container, { title, subtitle, empty, draw, table }) {
  const body = el("div", { className: "card-body" });
  const toggle = el("button", { type: "button", className: "card-toggle" });

  /** Fills the card body with the chart, the table or the empty message. */
  function show() {
    const asTable = showingTable.get(title) === true;
    toggle.textContent = asTable ? "Show chart" : "Show table";
    toggle.setAttribute("aria-pressed", String(asTable));
    if (empty) {
      body.replaceChildren(el("p", { className: "card-empty", textContent: "No bookings match these filters." }));
    } else if (asTable) {
      body.replaceChildren(el("div", { className: "table-scroll" }, dataTable(table.columns, table.rows)));
    } else {
      draw(body);
    }
  }

  toggle.onclick = () => {
    showingTable.set(title, !showingTable.get(title));
    show();
  };

  container.replaceChildren(
    el("header", { className: "card-header" },
      el("div", {},
        el("h2", { textContent: title }),
        el("p", { textContent: subtitle })),
      empty ? "" : toggle),
    body);
  show(); // after the card is on the page, so the chart can measure its width
}
