/**
 * @file Fills the page with the spa's facts from /api/spa: its name, a short
 * introduction, the opening hours and the contact details.
 *
 * Everything about the spa comes from the business document through the
 * server; nothing is hard-coded here.
 */

import { $, el } from "./dom.js";

/**
 * The opening sentences of a paragraph, for a short introduction.
 * @param {string} text e.g. the "About" paragraph
 * @param {number} count how many sentences to keep
 * @returns {string}
 */
function firstSentences(text, count) {
  // A sentence ends at ". ", "! " or "? ".
  return text.split(/(?<=[.!?])\s+/).slice(0, count).join(" ");
}

/**
 * Writes the spa's facts into the page.
 * @param {object} spa the /api/spa response (see KnowledgeBase.public() in spadesk/business/knowledge.py)
 */
export function renderSpa(spa) {
  document.title = spa.name;
  document.querySelectorAll("[data-spa-name]").forEach((node) => (node.textContent = spa.name));
  $("#about").textContent = firstSentences(spa.about, 2);

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
