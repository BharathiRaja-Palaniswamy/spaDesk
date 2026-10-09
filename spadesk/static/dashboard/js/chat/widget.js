/**
 * @file The chat window: a button in the corner that opens a small panel with
 * a message list and an input box. This file is only the window. What the
 * assistant says comes from askAssistant() in assistant.js.
 */

import { $, el } from "../dom.js";
import { askAssistant } from "./assistant.js";

/**
 * Wires up the chat button, panel and form that are in index.html.
 */
export function initChat() {
  const launcher = $("#chat-launcher");
  const panel = $("#chat-panel");
  const messages = $("#chat-messages");
  const form = $("#chat-form");
  const input = $("#chat-input");

  /**
   * Opens or closes the panel.
   * @param {boolean} open
   */
  function setOpen(open) {
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    if (open) input.focus();
  }

  /**
   * Adds one bubble to the message list.
   * @param {string} who CSS classes for the bubble: "user", "assistant" or "assistant pending"
   * @param {string} text
   * @returns {HTMLElement} the bubble, so its text can be changed later
   */
  function addMessage(who, text) {
    const bubble = el("div", { className: `chat-message ${who}`, textContent: text });
    messages.append(bubble);
    messages.scrollTop = messages.scrollHeight; // keep the newest message in view
    return bubble;
  }

  launcher.addEventListener("click", () => setOpen(panel.hidden));
  $("#chat-close").addEventListener("click", () => setOpen(false));
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setOpen(false);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault(); // stop the browser reloading the page
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    addMessage("user", question);

    const bubble = addMessage("assistant pending", "Thinking…");
    try {
      const reply = await askAssistant(question);
      // The answer, then its list when it has one. Each entry is a bullet:
      // the name in bold, the detail after it.
      bubble.replaceChildren(
        el("p", { textContent: reply.answer }),
        reply.items.length ? el("ul", { className: "chat-list" }, ...reply.items.map((item) =>
          el("li", {}, el("strong", { textContent: item.name }), ` ${item.detail}`))) : "");
      // The other fields: mark an answer the profile did not cover, and show
      // where a covered one came from.
      bubble.classList.toggle("not-covered", !reply.covered);
      if (reply.sections.length) {
        messages.append(el("div", { className: "chat-sources", textContent: `From: ${reply.sections.join(", ")}` }));
      }
    } catch {
      bubble.textContent = "Sorry, something went wrong. Please try again.";
    }
    bubble.classList.remove("pending");
    messages.scrollTop = messages.scrollHeight;
  });
}
