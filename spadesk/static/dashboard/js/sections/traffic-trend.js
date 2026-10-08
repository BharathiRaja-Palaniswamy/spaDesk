/**
 * @file The "Traffic trend" card: how many bookings were made over time.
 * "Traffic" here means bookings. The data has no website visits or enquiries.
 *
 * Data: report.trend, the same points the revenue trend uses.
 */

import { chartCard } from "../card.js";
import { drawLineChart } from "../charts/line-chart.js";
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
    draw: (body) => drawLineChart(body, {
      labels,
      series: [{ name: "Bookings", className: "series-1", values: points.map((point) => point.bookings) }],
      formatValue: number,
      formatTick: number,
      area: true,
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
