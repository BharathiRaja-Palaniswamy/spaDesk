/**
 * @file The chat's connection to the model. `askAssistant` sends the question
 * to the server route POST /api/chat (see spadesk/assistant/customer.py),
 * which calls the model and returns its reply as structured data.
 *
 * The model is called from the server, never from this file: anything in the
 * browser is visible to every visitor, including an API key.
 */

/**
 * Answers one question.
 *
 * @param {string} question what the visitor typed
 * @param {boolean} rag true to answer from retrieved chunks, false from the whole profile
 * @returns {Promise<{answer: string, items: {name: string, detail: string}[], covered: boolean,
 *   sections: string[], suggestions: string[], rag: boolean, chunks: number}>}
 *   `answer` is the text to show; `items` holds the entries when the answer
 *   is a list (empty otherwise); `covered` is false when the profile does not
 *   contain the answer; `sections` names the parts of the profile it came
 *   from; `suggestions` holds follow-up questions; `rag` and `chunks` say how
 *   the answer was produced
 */
export async function askAssistant(question, rag) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, rag }),
  });
  const body = await response.json();
  // On an error the server sends { detail } and no reply; throwing makes the
  // chat show its error message.
  if (!response.ok) throw new Error(body.detail);
  return body;
}
