/**
 * @file The "Revenue by category" card: which service categories earn the most.
 *
 * Data: report.categories = [{ category, revenue, bookings, completed,
 * completion_rate, average_rating }], highest revenue first.
 */

import { chartCard } from "../card.js";
import { drawBarList } from "../charts/bar-list.js";
import { money, number, percent, rating } from "../format.js";

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `categories`, `summary` and `currency`
 */
export function renderCategoryRevenue(container, report) {
  const rows = report.categories;

  chartCard(container, {
    title: "Revenue by category",
    subtitle: "Revenue from completed bookings in each service category",
    empty: report.summary.bookings === 0,
    draw: (body) => drawBarList(body, {
      className: "series-1",
      rows: rows.map((row) => ({
        label: row.category,
        value: row.revenue,
        text: money(row.revenue, report.currency),
        detail: `${number(row.bookings)} bookings, ${percent(row.completion_rate)} completed, rating ${rating(row.average_rating)}`,
      })),
    }),
    table: {
      columns: ["Category", "Revenue", "Bookings", "Completed", "Completion rate", "Average rating"],
      rows: rows.map((row) => [row.category, money(row.revenue, report.currency), number(row.bookings),
        number(row.completed), percent(row.completion_rate), rating(row.average_rating)]),
    },
  });
}
