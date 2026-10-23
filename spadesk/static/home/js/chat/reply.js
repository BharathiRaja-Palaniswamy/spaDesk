/**
 * @file Turns one structured reply into what the chat shows. The model
 * returns data (see AgentReply in spadesk/agent/prompt.py); this file
 * decides how each field looks.
 */

import { el } from "../dom.js";

/**
 * Fills a message bubble from a reply: the answer, then its list when it has
 * one, each entry as a bullet (name in bold, detail after it).
 * @param {HTMLElement} bubble the assistant's bubble, currently showing "Thinking…"
 * @param {{answer: string, items: {name: string, detail: string}[]}} reply
 */
export function showReply(bubble, reply) {
  bubble.replaceChildren(
    el("p", { textContent: reply.answer }),
    reply.items.length ? el("ul", { className: "chat-list" }, ...reply.items.map((item) =>
      el("li", {}, el("strong", { textContent: item.name }), ` ${item.detail}`))) : "");
}

/**
 * A row of suggested replies, shown as small buttons ("chips"). Clicking one
 * sends it, exactly as if it had been typed.
 * @param {string[]} questions
 * @param {(question: string) => void} onPick called with the question that was clicked
 * @returns {HTMLElement}
 */
export function suggestionChips(questions, onPick) {
  return el("div", { className: "chat-suggestions" }, ...questions.map((question) =>
    el("button", { type: "button", className: "chat-chip", textContent: question, onclick: () => onPick(question) })));
}
