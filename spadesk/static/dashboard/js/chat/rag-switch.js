/**
 * @file The "RAG on / RAG off" switch in the chat header. It decides how the
 * assistant is given the business profile for the next question:
 *
 *   off  the whole profile is sent to the model
 *   on   only the chunks retrieved for the question are sent (RAG)
 *
 * The switch only records the choice. The server acts on it: see `rag` in
 * ChatQuestion, spadesk/assistant/replies.py.
 */

import { $ } from "../dom.js";

/**
 * Wires up the switch that is in index.html and keeps its label in step.
 * @returns {() => boolean} a function that says whether RAG is on right now
 */
export function initRagSwitch() {
  const input = $("#chat-rag");
  const label = $("#chat-rag-label");
  input.addEventListener("change", () => {
    label.textContent = input.checked ? "RAG on" : "RAG off";
  });
  return () => input.checked;
}

/**
 * The words that say how one answer was produced, shown under it.
 * @param {{rag: boolean, chunks: number}} reply
 * @returns {string} e.g. "RAG · 3 chunks" or "Full profile"
 */
export const modeLabel = (reply) => (reply.rag ? `RAG · ${reply.chunks} chunks` : "Full profile");
