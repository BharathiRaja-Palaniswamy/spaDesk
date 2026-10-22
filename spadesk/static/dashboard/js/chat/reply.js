/**
 * @file Turns one structured reply from the assistant into what the chat
 * shows. The model returns data (see ChatReply in spadesk/reports/routes.py);
 * this file decides how each field looks.
 */

import { el } from "../dom.js";

/**
 * Fills a message bubble from a reply, top to bottom:
 *   1. the answer;
 *   2. its list when it has one, each entry as a bullet (name in bold, detail
 *      after it): services with prices, or the figures of a report;
 *   3. for a report answer, an "Insights" block: what the figures mean.
 * An answer the assistant could not cover gets a different style.
 * @param {HTMLElement} bubble the assistant's bubble, currently showing "Thinking…"
 * @param {{answer: string, items: {name: string, detail: string}[], insights: string[],
 *   covered: boolean}} reply
 */
export function showReply(bubble, reply) {
  bubble.replaceChildren(
    el("p", { textContent: reply.answer }),
    reply.items.length ? el("ul", { className: "chat-list" }, ...reply.items.map((item) =>
      el("li", {}, el("strong", { textContent: item.name }), ` ${item.detail}`))) : "",
    reply.insights.length ? el("div", { className: "chat-insights" },
      el("p", { className: "chat-insights-title", textContent: "Insights" }),
      el("ul", {}, ...reply.insights.map((insight) => el("li", { textContent: insight })))) : "");
  bubble.classList.toggle("not-covered", !reply.covered);
}

/**
 * The small line that goes under an answer: where it came from. E.g.
 * "From: Cancellation Policy", or "Report: 2026-09-01 to 2026-09-30" for an
 * answer made from report figures, or both.
 * @param {string[]} sections the profile headings the answer came from; may be empty
 * @param {string[]} reports the reports the assistant asked for; may be empty
 * @returns {HTMLElement}
 */
export function sourceLine(sections, reports) {
  const parts = [sections.length && `From: ${sections.join(", ")}`, reports.length && `Report: ${reports.join("; ")}`];
  return el("div", { className: "chat-sources", textContent: parts.filter(Boolean).join(" · ") });
}
