/**
 * @file The maths that turns data values into pixel positions.
 */

/**
 * Makes a function that maps a value from one range onto another.
 * @param {number} fromMin start of the data range
 * @param {number} fromMax end of the data range
 * @param {number} toMin pixel position for fromMin
 * @param {number} toMax pixel position for fromMax
 * @returns {(value: number) => number}
 * @example linear(0, 100, 200, 0)(50) // 100: value 0..100 onto pixels 200..0
 */
export function linear(fromMin, fromMax, toMin, toMax) {
  const span = fromMax - fromMin || 1; // avoid dividing by zero when all values are equal
  return (value) => toMin + ((value - fromMin) / span) * (toMax - toMin);
}

/**
 * Round axis values for a chart whose largest value is `max`. The last tick
 * is the top of the axis.
 * @param {number} max the largest value to fit
 * @param {number} [roughCount] about how many steps to aim for
 * @returns {number[]} e.g. 4870 -> [0, 1000, 2000, 3000, 4000, 5000]
 */
export function axisTicks(max, roughCount = 4) {
  if (max <= 0) return [0, 1];
  const rawStep = max / roughCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  // Pick the first "round" step that is at least as big as the raw one.
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rawStep);
  const ticks = [];
  for (let value = 0; value < max + step; value += step) ticks.push(Math.round(value * 1e6) / 1e6);
  return ticks;
}

/**
 * Up to `count` positions spread evenly across a list, always including the
 * first. Used to thin out crowded x-axis labels.
 * @param {number} length how many items the list has
 * @param {number} count the most positions to return
 * @returns {number[]} indexes into the list, e.g. (30, 6) -> [0, 6, 12, 17, 23, 29]
 */
export function spreadIndexes(length, count) {
  if (length <= count) return [...Array(length).keys()];
  const step = (length - 1) / (count - 1);
  return [...new Set(Array.from({ length: count }, (_, i) => Math.round(i * step)))];
}
