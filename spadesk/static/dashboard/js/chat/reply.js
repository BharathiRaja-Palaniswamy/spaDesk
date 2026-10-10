/**
 * @file Turns one structured reply from the assistant into what the chat
 * shows. The model returns data (see ChatReply in spadesk/reports/routes.py);
 * this file decides how each field looks.
 */

import { el } from "../dom.js";

/**
 * Fills a message bubble from a reply: the answer, then its list when it has
 * one, with each entry as a bullet (name in bold, detail after it). An answer
 * the business profile did not cover gets a different style.
 * @param {HTMLElement} bubble the assistant's bubble, currently showing "Thinking…"
 * @param {{answer: string, items: {name: string, detail: string}[], covered: boolean}} reply
 */
export function showReply(bubble, reply) {
  bubble.replaceChildren(
    el("p", { textContent: reply.answer }),
    reply.items.length ? el("ul", { className: "chat-list" }, ...reply.items.map((item) =>
      el("li", {}, el("strong", { textContent: item.name }), ` ${item.detail}`))) : "");
  bubble.classList.toggle("not-covered", !reply.covered);
}

/**
 * The "From: Cancellation Policy" line that goes under an answer.
 * @param {string[]} sections the profile headings the answer came from
 * @returns {HTMLElement|string} "" when there is nothing to cite
 */
export function sourceLine(sections) {
  return sections.length ? el("div", { className: "chat-sources", textContent: `From: ${sections.join(", ")}` }) : "";
}
