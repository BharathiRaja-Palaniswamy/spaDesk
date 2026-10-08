/**
 * @file The assistant's connection to the model. `askAssistant` sends the
 * question to the server route POST /api/report/chat (see
 * spadesk/reports/routes.py), which calls the model and returns its reply.
 *
 * The model is called from the server, never from this file: anything in the
 * browser is visible to every visitor, including an API key.
 */

/**
 * Answers one question. Called by the chat window (widget.js), which shows
 * "Thinking…" until the returned promise resolves, then shows the text. If
 * this throws, the window shows an error message instead.
 *
 * @param {string} question what the visitor typed
 * @param {?object} report the report currently on screen (the /api/report
 *   response), so the model can be given the numbers the visitor is looking
 *   at; null if no report has loaded yet
 * @returns {Promise<string>} the assistant's reply
 */
export async function askAssistant(question, report) {
  const response = await fetch("/api/report/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const body = await response.json();
  // On an error the server sends { detail } and no answer; throwing makes the
  // chat window show its error message.
  if (!response.ok) throw new Error(body.detail);
  return body.answer;
}
