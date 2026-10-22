/**
 * @file The assistant's connection to the model. `askAssistant` sends the
 * question to the server route POST /api/report/chat (see
 * spadesk/reports/routes.py), which calls the model and returns its reply as
 * structured data: not just text, but named fields the page can act on.
 *
 * The model is called from the server, never from this file: anything in the
 * browser is visible to every visitor, including an API key.
 */

/**
 * Answers one question. Called by the chat window (widget.js), which shows
 * "Thinking…" until the returned promise resolves, then shows the reply. If
 * this throws, the window shows an error message instead.
 *
 * @param {string} question what the visitor typed
 * @returns {Promise<{answer: string, items: {name: string, detail: string}[], insights: string[],
 *   covered: boolean, sections: string[], suggestions: string[], rag: boolean, chunks: number,
 *   reports: string[]}>}
 *   `answer` is the text to show; `items` holds the entries when the answer
 *   is a list or report figures (empty otherwise); `insights` holds what the
 *   figures mean, for a report answer; `covered` is false when the business profile
 *   does not contain the answer; `sections` names the parts of the profile
 *   the answer came from; `suggestions` holds follow-up questions to offer;
 *   `rag` and `chunks` say how the answer was produced; `reports` lists the
 *   report periods the assistant looked up, when the answer uses figures
 */
export async function askAssistant(question) {
  const response = await fetch("/api/report/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // The server answers from the chunks retrieved for the question (RAG).
    body: JSON.stringify({ question }),
  });
  const body = await response.json();
  // On an error the server sends { detail } and no reply; throwing makes the
  // chat window show its error message.
  if (!response.ok) throw new Error(body.detail);
  return body;
}
