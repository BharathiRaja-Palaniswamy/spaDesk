/**
 * @file The "Revenue by channel" card: what each source channel earned over
 * the whole selected period.
 *
 * Data: report.channels = [{ channel, revenue, bookings, completed,
 * completion_rate, average_rating }], in the data file's channel order.
 */

import { chartCard } from "../card.js";
import { drawBarList } from "../charts/bar-list.js";
import { money, number, percent, rating } from "../format.js";

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `channels`, `summary` and `currency`
 */
export function renderChannelRevenue(container, report) {
  const rows = report.channels;

  chartCard(container, {
    title: "Revenue by channel",
    subtitle: "Revenue from completed bookings that came through each channel",
    empty: report.summary.bookings === 0,
    draw: (body) => drawBarList(body, {
      // A channel's colour comes from its position in the data file's channel
      // list (series-1 to series-5), so it keeps that colour whatever the filters are.
      rows: rows.map((row, index) => ({
        label: row.channel,
        className: `series-${index + 1}`,
        value: row.revenue,
        text: money(row.revenue, report.currency),
        detail: `${number(row.bookings)} bookings, ${percent(row.completion_rate)} completed, rating ${rating(row.average_rating)}`,
      })),
    }),
    table: {
      columns: ["Channel", "Revenue", "Bookings", "Completed", "Completion rate", "Average rating"],
      rows: rows.map((row) => [row.channel, money(row.revenue, report.currency), number(row.bookings),
        number(row.completed), percent(row.completion_rate), rating(row.average_rating)]),
    },
  });
}
