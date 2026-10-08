/**
 * @file The "New and returning customers" card: of the people who booked in
 * the selected period, how many were new to the spa and how many had booked
 * before. It counts customers, not bookings.
 *
 * Data: report.customers = {
 *   customers,           how many different people booked in the period
 *   repeat_customers,    how many of them booked more than once in the period
 *   has_history,         false when the period starts where the data starts,
 *                        so nobody can be known to be returning
 *   by_type: [{ customer_type: "New" | "Returning", customers, bookings,
 *               revenue, completion_rate, average_rating }]
 * }
 * New = first ever booking is inside the period. Returning = first booked before it.
 */

import { chartCard } from "../card.js";
import { drawBarList } from "../charts/bar-list.js";
import { el } from "../dom.js";
import { money, number, percent, rating } from "../format.js";

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `customers` and `currency`
 */
export function renderCustomerMix(container, report) {
  const { customers: total, repeat_customers: repeat, has_history: hasHistory, by_type: types } = report.customers;

  chartCard(container, {
    title: "New and returning customers",
    subtitle: "Customers who first booked in this period, and those who had booked before it",
    empty: total === 0,
    draw: (body) => {
      const bars = el("div");
      body.replaceChildren(bars,
        el("p", { className: "card-note",
          textContent: `${number(repeat)} of these ${number(total)} customers booked more than once during this period.` }),
        // At the very start of the data everyone is "new" by definition; say so.
        hasHistory ? "" : el("p", { className: "card-note",
          textContent: "This period starts where the data starts, so there is no earlier history: every customer counts as new." }));
      drawBarList(bars, {
        className: "series-1",
        rows: types.map((row) => ({
          label: `${row.customer_type} customers`,
          value: row.customers,
          text: `${number(row.customers)} · ${percent(row.customers / total)}`,
          detail: `${number(row.bookings)} bookings, ${money(row.revenue, report.currency)} revenue, rating ${rating(row.average_rating)}`,
        })),
      });
    },
    table: {
      columns: ["Group", "Customers", "Share", "Bookings", "Revenue", "Completion rate", "Average rating"],
      rows: types.map((row) => [`${row.customer_type} customers`, number(row.customers), percent(row.customers / total),
        number(row.bookings), money(row.revenue, report.currency), percent(row.completion_rate), rating(row.average_rating)]),
    },
  });
}
