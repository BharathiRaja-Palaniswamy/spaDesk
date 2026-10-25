/**
 * @file Turns one structured reply into what the chat shows. The model
 * returns data (see AgentReply in spadesk/agent/pebby.py); this file
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

/**
 * Debug mode (SPADESK_DEBUG on, for testing): what happened between the
 * agents in this turn, folded under the reply. On top, the turn's path as a
 * flow (Customer → Pebby → booking → Collect → …); under it, one line per step
 * (who → whom, and what), with the data behind each step folded away.
 * Each actor is coloured by its kind: AI, small AI, code, tool, the customer.
 * @param {{who: string, to: string, kinds: string[], what: string, data: unknown}[]} steps
 * @returns {HTMLElement}
 */
export function debugPanel(steps) {
  const actor = (name, kind) => el("span", { className: `chat-debug-actor kind-${kind}`, textContent: name });
  const arrow = () => el("span", { className: "chat-debug-arrow", textContent: "→" });

  // The path: from the customer's message, through everyone who acted or was passed to, each once in a row.
  const path = [["Customer", "person"]];
  for (const step of steps) {
    if (path.at(-1)[0] !== step.who) path.push([step.who, step.kinds[0]]);
    if (path.at(-1)[0] !== step.to) path.push([step.to, step.kinds[1]]);
  }
  const key = el("div", { className: "chat-debug-key" },
    ...[["AI", "ai"], ["small AI", "small-ai"], ["code", "code"], ["tool", "tool"], ["customer", "person"]]
      .map(([name, kind]) => actor(name, kind)));
  const flow = el("div", { className: "chat-debug-flow" },
    ...path.flatMap(([name, kind], n) => (n ? [arrow(), actor(name, kind)] : [actor(name, kind)])));

  const lines = steps.map((step) => el("li", {},
    el("div", { className: "chat-debug-step" },
      actor(step.who, step.kinds[0]), arrow(), actor(step.to, step.kinds[1]),
      el("span", { className: "chat-debug-what", textContent: step.what })),
    step.data == null ? "" : el("details", { className: "chat-debug-data" },
      el("summary", { textContent: "data" }), el("pre", { textContent: JSON.stringify(step.data, null, 2) }))));

  return el("details", { className: "chat-debug" },
    el("summary", { textContent: `Debug: ${steps.length} step${steps.length === 1 ? "" : "s"}` }),
    flow, el("ol", {}, ...lines), key);
}
