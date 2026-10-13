/**
 * @file The chat window: a button in the corner that opens a small panel with
 * a message list and an input box. This file is only the window. What the
 * assistant says comes from askAssistant() in assistant.js, how a reply is
 * laid out from reply.js, the opening message from welcome.js, the
 * suggested-question buttons from suggestions.js, and the RAG switch from
 * rag-switch.js.
 */

import { $, el } from "../dom.js";
import { askAssistant } from "./assistant.js";
import { initRagSwitch, modeLabel } from "./rag-switch.js";
import { showReply, sourceLine } from "./reply.js";
import { suggestionChips } from "./suggestions.js";
import { introduction, STARTER_QUESTIONS } from "./welcome.js";

/**
 * Wires up the chat button, panel and form that are in index.html.
 */
export function initChat() {
  const launcher = $("#chat-launcher");
  const panel = $("#chat-panel");
  const messages = $("#chat-messages");
  const form = $("#chat-form");
  const input = $("#chat-input");
  /** Says whether the RAG switch in the header is on. */
  const ragIsOn = initRagSwitch();
  /** True while a reply is on its way, so two questions cannot overlap. */
  let busy = false;

  /**
   * Opens or closes the panel. The first time it opens, the assistant
   * introduces itself and offers a few questions.
   * @param {boolean} open
   */
  function setOpen(open) {
    panel.hidden = !open;
    launcher.setAttribute("aria-expanded", String(open));
    if (!open) return;
    if (!messages.hasChildNodes()) {
      addMessage("assistant", introduction($("#business-name").textContent.trim()));
      addSuggestions(STARTER_QUESTIONS);
    }
    input.focus();
  }

  /**
   * Adds one bubble to the message list.
   * @param {string} who CSS classes for the bubble: "user", "assistant" or "assistant pending"
   * @param {string} text
   * @returns {HTMLElement} the bubble, so its content can be changed later
   */
  function addMessage(who, text) {
    const bubble = el("div", { className: `chat-message ${who}`, textContent: text });
    messages.append(bubble);
    scrollToEnd();
    return bubble;
  }

  /**
   * Adds a row of suggested questions under the latest message.
   * @param {string[]} questions
   */
  function addSuggestions(questions) {
    if (questions.length) messages.append(suggestionChips(questions, ask));
    scrollToEnd();
  }

  /** Keeps the newest message in view. */
  function scrollToEnd() {
    messages.scrollTop = messages.scrollHeight;
  }

  /**
   * Asks one question and shows the reply. Used by the input box and by the
   * suggested-question buttons alike.
   * @param {string} question
   */
  async function ask(question) {
    if (busy || !question) return;
    busy = true;
    // Earlier suggestions are removed once a question is asked: they belonged
    // to the previous answer, and would otherwise pile up in the conversation.
    messages.querySelectorAll(".chat-suggestions").forEach((row) => row.remove());
    addMessage("user", question);

    const bubble = addMessage("assistant pending", "Thinking…");
    try {
      const reply = await askAssistant(question, ragIsOn());
      // Each field of the structured reply has its own place: the answer and
      // list in the bubble, the sources under it, then the follow-up questions.
      showReply(bubble, reply);
      messages.append(sourceLine(modeLabel(reply), reply.sections));
      addSuggestions(reply.suggestions);
    } catch {
      bubble.textContent = "Sorry, something went wrong. Please try again.";
    }
    bubble.classList.remove("pending");
    scrollToEnd();
    busy = false;
  }

  launcher.addEventListener("click", () => setOpen(panel.hidden));
  $("#chat-close").addEventListener("click", () => setOpen(false));
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setOpen(false);
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault(); // stop the browser reloading the page
    const question = input.value.trim();
    input.value = "";
    ask(question);
  });
}
