/**
 * @file Turns raw numbers and dates from the API into text for the page.
 * Nothing here knows about the page itself. Dates are "YYYY-MM-DD" strings
 * and are always read as UTC, so they never shift a day in the visitor's
 * time zone.
 */

/**
 * @param {number} value
 * @param {string} currency the symbol to put in front, e.g. "$"
 * @returns {string} 19818 -> "$19,818"; 105.41 -> "$105.41"
 */
export function money(value, currency) {
  const digits = Number.isInteger(value) ? 0 : 2;
  return currency + value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/**
 * Short money for axis labels.
 * @param {number} value
 * @param {string} currency
 * @returns {string} 12500 -> "$12.5K"; 800 -> "$800"
 */
export function compactMoney(value, currency) {
  return currency + new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

/**
 * @param {number} value
 * @returns {string} 2490 -> "2,490"
 */
export const number = (value) => value.toLocaleString("en-US");

/**
 * @param {number} fraction a share between 0 and 1
 * @returns {string} 0.8624 -> "86.2%"
 */
export const percent = (fraction) => `${(fraction * 100).toFixed(1)}%`;

/**
 * @param {?number} value an average rating, or null when nobody rated
 * @returns {string} 4.3 -> "4.30"; null -> "–"
 */
export const rating = (value) => (value === null ? "–" : value.toFixed(2));

/**
 * @param {string} iso "2026-09-01"
 * @returns {string} "1 Sep 2026"
 */
function dateLabel(iso) {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
  });
}

/**
 * @param {string} start ISO date
 * @param {string} end ISO date
 * @returns {string} "1 Sep 2026 – 30 Sep 2026"
 */
export const dateRange = (start, end) => `${dateLabel(start)} – ${dateLabel(end)}`;

/**
 * Number of days from start to end, counting both.
 * @param {string} start ISO date
 * @param {string} end ISO date
 * @returns {number} 1 Sep to 30 Sep -> 30
 */
export function dayCount(start, end) {
  return Math.round((new Date(`${end}T00:00:00Z`) - new Date(`${start}T00:00:00Z`)) / 86400000) + 1;
}

/**
 * The label for one point on a trend.
 * @param {string} iso the first day of the period
 * @param {"day"|"week"|"month"} bucket how the trend is sliced
 * @returns {string} a day or week by its date ("1 Sep"), a month by its name ("Sep 2026")
 */
export function bucketLabel(iso, bucket) {
  const options = bucket === "month" ? { month: "short", year: "numeric" } : { day: "numeric", month: "short" };
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", { ...options, timeZone: "UTC" });
}
