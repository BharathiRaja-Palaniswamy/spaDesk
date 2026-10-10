/**
 * @file A row of suggested questions, shown as small buttons ("chips").
 * Clicking one asks it, exactly as if it had been typed.
 */

import { el } from "../dom.js";

/**
 * Builds a row of question chips.
 * @param {string[]} questions
 * @param {(question: string) => void} onPick called with the question that was clicked
 * @returns {HTMLElement}
 */
export function suggestionChips(questions, onPick) {
  return el("div", { className: "chat-suggestions" }, ...questions.map((question) =>
    el("button", { type: "button", className: "chat-chip", textContent: question, onclick: () => onPick(question) })));
}
