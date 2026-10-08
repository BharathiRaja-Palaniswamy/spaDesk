/**
 * @file The chat window: a button in the corner that opens a small panel with
 * a message list and an input box. This file is only the window. What the
 * assistant says comes from askAssistant() in assistant.js.
 */

import { $, el } from "../dom.js";
import { askAssistant } from "./assistant.js";

/**
 * Wires up the chat button, panel and form that are in index.html.
 * @param {() => ?object} getReport returns the report currently on screen; it
 *   is passed along with each question
 */
export function initChat(getReport) {
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

    const reply = addMessage("assistant pending", "Thinking…");
    try {
      reply.textContent = await askAssistant(question, getReport());
    } catch {
      reply.textContent = "Sorry, something went wrong. Please try again.";
    }
    reply.classList.remove("pending");
    messages.scrollTop = messages.scrollHeight;
  });
}
