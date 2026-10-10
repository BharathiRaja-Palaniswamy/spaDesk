/**
 * @file The "Traffic trend" card: how many bookings were made over time, one
 * bar per period.
 * "Traffic" here means bookings. The data has no website visits or enquiries.
 *
 * Data: report.trend, the same points the revenue trend uses.
 */

import { chartCard } from "../card.js";
import { drawColumnChart } from "../charts/column-chart.js";
import { timeAxis } from "../charts/time-axis.js";
import { bucketLabel, number, percent } from "../format.js";

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `trend` and `summary`
 */
export function renderTrafficTrend(container, report) {
  const { bucket, points } = report.trend;
  const labels = points.map((point) => bucketLabel(point.start, bucket));

  chartCard(container, {
    title: "Traffic trend",
    subtitle: `Bookings made, per ${bucket}`,
    empty: report.summary.bookings === 0,
    draw: (body) => drawColumnChart(body, {
      labels,
      axis: timeAxis(points.map((point) => point.start), bucket),
      values: points.map((point) => point.bookings),
      name: "Bookings",
      className: "series-1",
      formatValue: number,
      formatTick: number,
    }),
    table: {
      columns: [bucket === "day" ? "Day" : `${bucket[0].toUpperCase()}${bucket.slice(1)} starting`,
        "Bookings", "Completed", "Completion rate"],
      rows: points.map((point, index) => [
        labels[index], number(point.bookings), number(point.completed), percent(point.completion_rate),
      ]),
    },
  });
}
