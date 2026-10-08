/**
 * @file Small helpers for finding and building page elements.
 */

/**
 * Finds the first element matching a CSS selector.
 * @param {string} selector e.g. "#tiles" or ".card"
 * @returns {?Element}
 */
export const $ = (selector) => document.querySelector(selector);

/**
 * Builds an element.
 *
 * Text is always set with textContent, never innerHTML, so a value from the
 * data file can never be run as HTML.
 *
 * @param {string} tag e.g. "p"
 * @param {object} [props] properties to set, e.g. { className: "note", textContent: "hi" }
 * @param {...(Node|string|null)} children added inside, in order; null and "" are skipped
 * @returns {HTMLElement}
 * @example el("p", { className: "note" }, "Total: ", el("strong", { textContent: "12" }))
 */
export function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((child) => child !== null && child !== ""));
  return node;
}
