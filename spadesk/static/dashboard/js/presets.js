/**
 * @file The date-range shortcuts in the filter row.
 *
 * Each preset turns the data's first and last day into a { start, end } pair.
 * Presets count back from the LAST DAY IN THE DATA, not from today: the data
 * is a fixed file, so "last 30 days" means its most recent 30 days.
 */

/**
 * Moves an ISO date by a number of days.
 * @param {string} iso "2026-09-30"
 * @param {number} days may be negative
 * @returns {string} the new ISO date
 */
function addDays(iso, days) {
  const date = new Date(`${iso}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

/**
 * The later of two ISO dates. Text comparison works because "YYYY-MM-DD"
 * sorts the same way as the calendar.
 * @param {string} a
 * @param {string} b
 * @returns {string}
 */
const later = (a, b) => (a > b ? a : b);

/**
 * Makes the range function for "the last N days of the data".
 * @param {number} days
 * @returns {(min: string, max: string) => {start: string, end: string}}
 */
const lastDays = (days) => (min, max) => ({ start: later(min, addDays(max, -(days - 1))), end: max });

/**
 * The preset buttons, in the order shown. `range(min, max)` receives the first
 * and last date in the data and returns the dates to select.
 * @type {{id: string, label: string, range: (min: string, max: string) => {start: string, end: string}}[]}
 */
export const PRESETS = [
  { id: "7d", label: "Last 7 days", range: lastDays(7) },
  { id: "30d", label: "Last 30 days", range: lastDays(30) },
  { id: "90d", label: "Last 90 days", range: lastDays(90) },
  // The calendar month the data ends in.
  { id: "month", label: "Latest month", range: (min, max) => ({ start: later(min, `${max.slice(0, 8)}01`), end: max }) },
  { id: "all", label: "All data", range: (min, max) => ({ start: min, end: max }) },
];

/** The preset selected when the page opens. */
export const DEFAULT_PRESET = "30d";
