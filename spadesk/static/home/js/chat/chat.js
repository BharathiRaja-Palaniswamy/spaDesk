/**
 * @file The conversation in the chat window: a message list and an input
 * box. Opening and closing the window is window.js. What the assistant says
 * comes from askAssistant() in assistant.js, how a reply is laid out from
 * reply.js, and the greeting from welcome.js.
 */

import { $, el } from "../dom.js";
import { askAssistant } from "./assistant.js";
import { showReply, sourceLine, suggestionChips } from "./reply.js";
import { greeting, STARTER_QUESTIONS } from "./welcome.js";

/**
 * Wires up the chat that is in index.html and greets the visitor.
 * @param {string} business the spa's name, used in the greeting
 */
export function initChat(business) {
  const messages = $("#chat-messages");
  const form = $("#chat-form");
  const input = $("#chat-input");
  /** True while a reply is on its way, so two questions cannot overlap. */
  let busy = false;

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
    // Earlier suggestions belonged to the previous answer; remove them so
    // they do not pile up in the conversation.
    messages.querySelectorAll(".chat-suggestions").forEach((row) => row.remove());
    addMessage("user", question);

    const bubble = addMessage("assistant pending", "Thinking…");
    try {
      const reply = await askAssistant(question);
      // Each field of the structured reply has its own place: the answer and
      // list in the bubble, the sources under it, then the follow-up questions.
      showReply(bubble, reply);
      if (reply.sections.length) messages.append(sourceLine(reply.sections));
      addSuggestions(reply.suggestions);
    } catch {
      bubble.textContent = "Sorry, something went wrong. Please try again.";
    }
    bubble.classList.remove("pending");
    scrollToEnd();
    busy = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault(); // stop the browser reloading the page
    const question = input.value.trim();
    input.value = "";
    ask(question);
  });

  addMessage("assistant", greeting(business));
  addSuggestions(STARTER_QUESTIONS);
}
