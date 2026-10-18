/**
 * @file Fills the page with the spa's facts from /api/spa: its name, the
 * opening hours and the contact details.
 *
 * Everything about the spa comes from the business document through the
 * server; nothing is hard-coded here. (The hero's welcome line is page
 * wording, written in index.html; the spa's own introduction is in the
 * About band.)
 */

import { $, el } from "./dom.js";

/**
 * Writes the spa's facts into the page.
 * @param {object} spa the /api/spa response (see KnowledgeBase.public() in spadesk/business/knowledge.py)
 */
export function renderSpa(spa) {
  document.title = spa.name;
  document.querySelectorAll("[data-spa-name]").forEach((node) => (node.textContent = spa.name));

  $("#hours").replaceChildren(...spa.hours.map((h) =>
    el("li", {}, el("span", { textContent: h.days }), el("span", { textContent: h.times }))));

  // Each fact is a label and a value, e.g. "Phone" and "(408) 555-0188".
  $("#contact").replaceChildren(...Object.entries(spa.contact).map(([label, value]) =>
    el("div", {}, el("dt", { textContent: label }), el("dd", { textContent: value }))));
}

/**
 * Loads the spa's facts from the server.
 * @returns {Promise<object>} the /api/spa response
 */
export async function loadSpa() {
  const response = await fetch("/api/spa");
  return response.json();
}
