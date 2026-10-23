/**
 * @file The "Powered by Claude" line under the chat: which provider answers,
 * read from the server (GET /api/powered-by), so it follows the app's
 * SPADESK_MODELS setting. If it cannot be read, the line stays hidden.
 */

import { $ } from "../dom.js";

/** Fills in and shows the line. */
export async function showPoweredBy() {
  const line = $("#chat-powered");
  try {
    const { customer } = await (await fetch("/api/powered-by")).json();
    line.textContent = `Powered by ${customer}`;
    line.hidden = false;
  } catch {
    line.hidden = true;
  }
}
