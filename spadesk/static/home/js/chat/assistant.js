/**
 * @file The chat's connection to Pebby, the booking agent. `askAssistant`
 * sends the customer's message to POST /api/assistant (see
 * spadesk/agent/routes.py), which runs one turn of the conversation on the
 * server and returns Pebby's reply.
 *
 * Every message carries this chat's thread id, made up when the page opens:
 * the server keeps the conversation under it, which is how Pebby remembers
 * the customer's name, what was said, and a booking waiting for their yes.
 * A new page is a new conversation.
 *
 * The model is called from the server, never from this file: anything in the
 * browser is visible to every visitor, including an API key.
 */

const THREAD = crypto.randomUUID();

/**
 * Sends one message and returns Pebby's reply.
 *
 * @param {string} message what the customer typed or tapped
 * @returns {Promise<{answer: string, items: {name: string, detail: string}[], suggestions: string[],
 *   confirming: boolean}>}
 *   `answer` is the text to show; `items` holds the entries when the answer is
 *   a list (empty otherwise); `suggestions` holds replies the customer can tap;
 *   `confirming` is true when Pebby is asking them to confirm a booking or
 *   cancellation, and the suggestions are then its yes and no
 */
export async function askAssistant(message) {
  const response = await fetch("/api/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread: THREAD, message }),
  });
  const body = await response.json();
  // On an error the server sends { detail } and no reply; throwing makes the
  // chat show its error message.
  if (!response.ok) throw new Error(body.detail);
  return body;
}
