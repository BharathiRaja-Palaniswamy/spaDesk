/**
 * @file The "Business profile" tab: a read-only view of the business
 * document, section by section.
 *
 * Data: GET /api/profile =
 *   { name, sections: [{ title, groups: [{ title, lines, bulleted, items }] }] }
 * where an item is { name, description, duration, price, lines }.
 * (spadesk/business/profile.py builds it and describes the shape in full.)
 */

import { getProfile } from "./api.js";
import { el } from "./dom.js";
import { dataTable } from "./table.js";

/** True once the profile has been fetched and drawn, so it is only loaded once. */
let loaded = false;

/**
 * Fetches the profile and draws it. Called each time the tab is shown; after
 * the first successful call it does nothing.
 * @param {Element} container the #profile box
 */
export async function showProfile(container) {
  if (loaded) return;
  try {
    const profile = await getProfile();
    container.replaceChildren(...profile.sections.map(sectionCard));
    loaded = true;
  } catch (error) {
    container.replaceChildren(el("p", { className: "panel-empty", textContent: error.message }));
  }
}

/**
 * One card per top-level heading of the document.
 * @param {{title: string, groups: object[]}} section
 * @returns {HTMLElement}
 */
function sectionCard(section) {
  // A section with a price table or several groups needs the full width.
  const wide = section.groups.length > 1 || section.groups.some((group) => group.items.some((item) => item.price));
  return el("section", { className: `card ${wide ? "wide" : ""}` },
    el("h2", { className: "profile-title", textContent: section.title }),
    ...section.groups.map(groupBlock));
}

/**
 * One group: an optional sub-heading, its text, then its entries.
 * @param {{title: string, lines: string[], bulleted: boolean, items: object[]}} group
 * @returns {HTMLElement}
 */
function groupBlock(group) {
  const priced = group.items.filter((item) => item.price);
  const others = group.items.filter((item) => !item.price);
  return el("div", { className: "profile-group" },
    group.title ? el("h3", { textContent: group.title }) : "",
    textBlock(group.lines, group.bulleted),
    priced.length ? priceTable(priced) : "",
    ...others.map((item) => el("div", { className: "profile-entry" },
      el("h4", { textContent: item.name }),
      textBlock(item.lines, item.lines.length > 1))));
}

/**
 * Lines of text as a bullet list or as paragraphs.
 * @param {string[]} lines
 * @param {boolean} bulleted
 * @returns {HTMLElement|string} "" when there are no lines
 */
function textBlock(lines, bulleted) {
  if (!lines.length) return "";
  return bulleted
    ? el("ul", { className: "profile-list" }, ...lines.map((line) => el("li", { textContent: line })))
    : el("div", { className: "profile-text" }, ...lines.map((line) => el("p", { textContent: line })));
}

/**
 * Services or packages as a table. The Duration column is left out when none
 * of the entries has one (packages, for example).
 * @param {{name: string, description: string, duration: string, price: string}[]} items
 * @returns {HTMLElement}
 */
function priceTable(items) {
  const hasDuration = items.some((item) => item.duration);
  const columns = ["Name", "Details", ...(hasDuration ? ["Duration"] : []), "Price"];
  const rows = items.map((item) => [item.name, item.description, ...(hasDuration ? [item.duration || "–"] : []), item.price]);
  return el("div", { className: "profile-table" }, dataTable(columns, rows));
}
