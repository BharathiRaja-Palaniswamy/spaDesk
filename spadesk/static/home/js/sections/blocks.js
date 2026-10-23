/**
 * @file Builds the page elements for one group of the business document,
 * chosen by the shape of its content rather than its name, so another
 * business's document is laid out just as well:
 *
 *   priced entries             cards with description, duration and price
 *   text entries (FAQ, offer)  cards, or questions that open on click
 *   "Lip Wax — $18" lines      a price list
 *   "Phone: (408) …" lines     a list of labels and values
 *   other lines                a bullet list or paragraphs, as written
 *
 * A group is { title, lines, bulleted, items }: see build_profile() in
 * spadesk/business/profile.py.
 */

import { el } from "../dom.js";
import { openBooking } from "../booking.js";

// "Lip Wax — $18" -> ["Lip Wax", "$18"]
const PRICED_LINE = /^(.+?)\s+[—–-]\s+(\$.+)$/;
// "Phone: (408) 555-0188" -> ["Phone", "(408) 555-0188"]. A short label only,
// so an ordinary sentence with a colon is not mistaken for one.
const LABELLED_LINE = /^([^:]{1,24}):\s+(.+)$/;

/**
 * A "Book" link that opens the booking pop-up with this service picked,
 * or nothing when the service cannot be booked.
 * @param {string} name
 * @param {Set<string>} bookable the services the booking form offers
 * @returns {?HTMLElement}
 */
function bookLink(name, bookable) {
  if (!bookable.has(name)) return null;
  return el("button", { type: "button", className: "book-link", textContent: "Book", onclick: () => openBooking(name) });
}

/**
 * One priced entry: a service or a package.
 * @param {{name: string, description: string, duration: string, price: string}} item
 * @param {Set<string>} bookable
 * @returns {HTMLElement}
 */
function pricedCard(item, bookable) {
  return el("article", { className: "card" },
    el("h4", { textContent: item.name }),
    item.description ? el("p", { textContent: item.description }) : null,
    el("div", { className: "card-meta" },
      el("span", { textContent: item.duration }),
      el("strong", { textContent: item.price })),
    bookLink(item.name, bookable));
}

/**
 * The text of an entry that is not priced: its own lines, or its description.
 * @param {{description: string, lines: string[]}} item
 * @returns {HTMLElement[]}
 */
function entryText(item) {
  const lines = item.lines.length ? item.lines : [item.description];
  return lines.length > 1
    ? [el("ul", {}, ...lines.map((line) => el("li", { textContent: line })))]
    : [el("p", { textContent: lines[0] })];
}

/**
 * Something that opens on click to show more: a question and its answer.
 * Built on <details>, which the browser opens and closes by itself.
 * @param {string} title shown while closed
 * @param {HTMLElement[]} body shown once opened
 * @returns {HTMLElement}
 */
export function foldout(title, body) {
  return el("details", { className: "foldout" }, el("summary", { textContent: title }), el("div", {}, ...body));
}

/**
 * The lines of a group, as the shape of the text suggests.
 * @param {string[]} lines
 * @param {boolean} bulleted whether they were a bullet list in the document
 * @param {Set<string>} bookable
 * @returns {?HTMLElement}
 */
function linesBlock(lines, bulleted, bookable) {
  if (!lines.length) return null;
  if (lines.every((line) => PRICED_LINE.test(line))) {
    return el("ul", { className: "price-list" }, ...lines.map((line) => {
      const [, name, price] = line.match(PRICED_LINE);
      return el("li", {}, el("span", { textContent: name }), bookLink(name, bookable), el("strong", { textContent: price }));
    }));
  }
  if (lines.every((line) => LABELLED_LINE.test(line))) {
    return el("dl", { className: "facts" }, ...lines.map((line) => {
      const [, label, value] = line.match(LABELLED_LINE);
      return el("div", {}, el("dt", { textContent: label }), el("dd", { textContent: value }));
    }));
  }
  return bulleted
    ? el("ul", { className: "text-list" }, ...lines.map((line) => el("li", { textContent: line })))
    : el("div", { className: "prose" }, ...lines.map((line) => el("p", { textContent: line })));
}

/**
 * Everything in one group: its lines, then its entries.
 * @param {{lines: string[], bulleted: boolean, items: object[]}} group
 * @param {"cards"|"accordion"} layout how text entries are shown
 * @param {Set<string>} bookable
 * @returns {HTMLElement}
 */
export function groupBlock(group, layout, bookable) {
  const priced = group.items.filter((item) => item.price);
  const text = group.items.filter((item) => !item.price);
  return el("div", { className: "group" },
    linesBlock(group.lines, group.bulleted, bookable),
    priced.length ? el("div", { className: "cards" }, ...priced.map((item) => pricedCard(item, bookable))) : null,
    text.length && layout === "accordion"
      ? el("div", { className: "foldouts" }, ...text.map((item) => foldout(item.name, entryText(item))))
      : null,
    text.length && layout === "cards"
      ? el("div", { className: "cards" }, ...text.map((item) =>
        el("article", { className: "card" }, el("h4", { textContent: item.name }), ...entryText(item))))
      : null);
}
