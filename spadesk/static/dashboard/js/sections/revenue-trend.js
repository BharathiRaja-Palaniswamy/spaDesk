/**
 * @file The "Revenue trend" card: revenue over time for the selected filters,
 * one bar per period.
 *
 * Data: report.trend = { bucket, points: [{ start, revenue, bookings, completed, ... }] },
 * one point per day, week or month (the server chooses, see metrics/period.py).
 */

import { chartCard } from "../card.js";
import { drawColumnChart } from "../charts/column-chart.js";
import { timeAxis } from "../charts/time-axis.js";
import { bucketLabel, compactMoney, money, number } from "../format.js";

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `trend`, `summary` and `currency`
 */
export function renderRevenueTrend(container, report) {
  const { bucket, points } = report.trend;
  const labels = points.map((point) => bucketLabel(point.start, bucket));

  chartCard(container, {
    title: "Revenue trend",
    subtitle: `Revenue from completed bookings, per ${bucket}`,
    empty: report.summary.bookings === 0,
    draw: (body) => drawColumnChart(body, {
      labels,
      axis: timeAxis(points.map((point) => point.start), bucket),
      values: points.map((point) => point.revenue),
      name: "Revenue",
      className: "series-1",
      formatValue: (value) => money(value, report.currency),
      formatTick: (value) => compactMoney(value, report.currency),
    }),
    table: {
      columns: [bucket === "day" ? "Day" : `${bucket[0].toUpperCase()}${bucket.slice(1)} starting`, "Revenue", "Bookings", "Completed"],
      rows: points.map((point, index) => [
        labels[index], money(point.revenue, report.currency), number(point.bookings), number(point.completed),
      ]),
    },
  });
}
