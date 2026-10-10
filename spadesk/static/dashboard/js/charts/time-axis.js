/**
 * @file Works out the labels for a chart's time axis, in two rows:
 *
 *   row 1   a short label under every bar:   24  25  26  27  28  29  30
 *   row 2   what those bars belong to:       September 2026
 *
 * Splitting the label this way keeps row 1 short enough that every bar can
 * have one. A single row of full dates ("27 Sept") does not fit under 30
 * bars, and leaving some out makes it look as if days are missing.
 */

/** A date read as UTC, so it never shifts a day in the visitor's time zone. */
const utc = (iso) => new Date(`${iso}T00:00:00Z`);
const text = (iso, options) => utc(iso).toLocaleDateString("en-GB", { ...options, timeZone: "UTC" });

/**
 * @param {string[]} starts the first day of each period, as ISO dates, in order
 * @param {"day"|"week"|"month"} bucket how the periods are sliced
 * @returns {{ticks: string[], groups: {label: string, from: number, to: number}[]}}
 *   `ticks` has one short label per period: the day of the month for days and
 *   weeks ("24"), the month for months ("Sep"). `groups` says which run of
 *   periods (index `from` to index `to`) shares the same month or year, and
 *   what to call it ("September 2026", or "2026").
 */
export function timeAxis(starts, bucket) {
  const byMonth = bucket === "month";
  const ticks = starts.map((iso) => (byMonth ? text(iso, { month: "short" }) : String(utc(iso).getUTCDate())));
  const groupOf = (iso) => (byMonth ? text(iso, { year: "numeric" }) : text(iso, { month: "long", year: "numeric" }));

  const groups = [];
  starts.forEach((iso, index) => {
    const label = groupOf(iso);
    const last = groups.at(-1);
    if (last && last.label === label) last.to = index; // same month (or year) as the bar before
    else groups.push({ label, from: index, to: index });
  });
  return { ticks, groups };
}
