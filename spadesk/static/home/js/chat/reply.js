/**
 * @file Turns one structured reply into what the chat shows. The model
 * returns data (see CustomerReply in spadesk/assistant/customer.py); this
 * file decides how each field looks.
 */

import { el } from "../dom.js";

/**
 * Fills a message bubble from a reply: the answer, then its list when it has
 * one, each entry as a bullet (name in bold, detail after it). An answer the
 * profile does not cover gets a different style.
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
 * The small line under an answer that says where it came from,
 * e.g. "From: Cancellation Policy".
 * @param {string[]} sections the profile headings the answer came from
 * @returns {HTMLElement}
 */
export function sourceLine(sections) {
  return el("div", { className: "chat-sources", textContent: `From: ${sections.join(", ")}` });
}

/**
 * A row of suggested questions, shown as small buttons ("chips"). Clicking
 * one asks it, exactly as if it had been typed.
 * @param {string[]} questions
 * @param {(question: string) => void} onPick called with the question that was clicked
 * @returns {HTMLElement}
 */
export function suggestionChips(questions, onPick) {
  return el("div", { className: "chat-suggestions" }, ...questions.map((question) =>
    el("button", { type: "button", className: "chat-chip", textContent: question, onclick: () => onPick(question) })));
}
