/**
 * @file The "Top services" card: the individual services that earn the most.
 * The chart shows the top few; the table view lists every service booked.
 *
 * Data: report.services = [{ service, category, price, revenue, bookings,
 * completed, completion_rate, average_rating }], highest revenue first.
 */

import { chartCard } from "../card.js";
import { drawBarList } from "../charts/bar-list.js";
import { money, number, percent, rating } from "../format.js";

/** How many services the chart shows. The table view has all of them. */
const TOP = 8;

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `services`, `summary` and `currency`
 */
export function renderTopServices(container, report) {
  const rows = report.services;

  chartCard(container, {
    title: "Top services",
    subtitle: rows.length > TOP
      ? `The ${TOP} services with the most revenue, of ${rows.length} booked`
      : "Revenue from completed bookings of each service",
    empty: report.summary.bookings === 0,
    draw: (body) => drawBarList(body, {
      className: "series-1",
      rows: rows.slice(0, TOP).map((row) => ({
        label: row.service,
        value: row.revenue,
        text: money(row.revenue, report.currency),
        detail: `${row.category}, ${number(row.bookings)} bookings, ${percent(row.completion_rate)} completed, rating ${rating(row.average_rating)}`,
      })),
    }),
    table: {
      columns: ["Service", "Category", "List price", "Revenue", "Bookings", "Completion rate", "Average rating"],
      rows: rows.map((row) => [row.service, row.category, money(row.price, report.currency),
        money(row.revenue, report.currency), number(row.bookings), percent(row.completion_rate), rating(row.average_rating)]),
    },
  });
}
