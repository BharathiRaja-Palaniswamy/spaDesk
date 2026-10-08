/**
 * @file The headline tiles at the top of the dashboard: a label, one big
 * number, and how it moved against the previous period.
 */

import { el } from "./dom.js";
import { dateRange, dayCount, money, number, percent, rating } from "./format.js";

/**
 * Which numbers to show and how. Adding a tile is adding one line here.
 *
 * - `key`      a field of the report's `summary`
 * - `format`   how to print the value
 * - `compare`  how to print the change against the previous period:
 *              "relative" = percent change (revenue: 2.1%),
 *              "points" = difference in percentage points (a rate: 5.9 pts),
 *              "difference" = plain difference (a rating: 0.27)
 * - `upIsGood` whether a rise is good news (it is not for lost revenue)
 *
 * @type {{key: string, label: string, format: Function, compare: string, upIsGood: boolean}[]}
 */
const TILES = [
  { key: "revenue", label: "Revenue", format: money, compare: "relative", upIsGood: true },
  { key: "bookings", label: "Bookings", format: number, compare: "relative", upIsGood: true },
  { key: "completion_rate", label: "Completion rate", format: percent, compare: "points", upIsGood: true },
  { key: "average_ticket", label: "Average ticket", format: money, compare: "relative", upIsGood: true },
  { key: "average_rating", label: "Average rating", format: rating, compare: "difference", upIsGood: true },
  { key: "lost_revenue", label: "Lost to cancellations and no-shows", format: money, compare: "relative", upIsGood: false },
];

/**
 * Draws all tiles.
 * @param {Element} container the #tiles box
 * @param {object} report uses `summary`, `previous_summary`, `previous_period`, `change`, `currency`
 */
export function renderTiles(container, report) {
  container.replaceChildren(...TILES.map((tile) => {
    const value = report.summary[tile.key];
    return el("article", { className: "tile" },
      el("span", { className: "tile-label", textContent: tile.label }),
      el("span", { className: "tile-value", textContent: tile.format(value, report.currency) }),
      deltaLine(tile, report));
  }));
}

/**
 * Builds the "▲ 2.1%" line under a tile's value, or a note when there is
 * nothing to compare with. What it is compared against ("Compared to previous
 * 30 days" and the exact dates) appears as a tooltip on hover or keyboard focus.
 * @param {object} tile one entry of TILES
 * @param {object} report
 * @returns {HTMLElement}
 */
function deltaLine(tile, report) {
  const amount = delta(tile, report);
  if (amount === null) {
    return el("span", { className: "tile-delta", textContent: "No earlier period to compare" });
  }
  const before = report.previous_period;
  const days = dayCount(before.start, before.end);
  const line = el("span", { className: "tile-delta has-tooltip", tabIndex: 0 });
  // css/tooltip.css shows this attribute as the tooltip text.
  line.dataset.tooltip = `Compared to previous ${days === 1 ? "day" : `${days} days`} (${dateRange(before.start, before.end)})`;

  if (amount.value === 0) {
    line.textContent = "No change";
    return line;
  }
  const up = amount.value > 0;
  line.append(el("span", {
    className: up === tile.upIsGood ? "good" : "bad", // green or red, see css/tiles.css
    textContent: `${up ? "▲" : "▼"} ${amount.text}`,
  }));
  return line;
}

/**
 * Works out how much one tile's number moved.
 * @param {object} tile one entry of TILES
 * @param {object} report
 * @returns {?{value: number, text: string}} `value` is signed (its sign gives the
 *   direction), `text` is the amount to print; null when it cannot be compared
 */
function delta(tile, report) {
  const before = report.previous_summary?.[tile.key];
  const now = report.summary[tile.key];
  if (before === null || before === undefined || now === null) return null;

  if (tile.compare === "points") {
    const points = (now - before) * 100;
    return { value: points, text: `${Math.abs(points).toFixed(1)} pts` };
  }
  if (tile.compare === "difference") {
    const difference = now - before;
    return { value: difference, text: Math.abs(difference).toFixed(2) };
  }
  const change = report.change[tile.key]; // worked out by the server
  return change === null ? null : { value: change, text: percent(Math.abs(change)) };
}
