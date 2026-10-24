/**
 * @file The "Powered by Claude" line under a chat, shared by the customer
 * site and the manage page: which provider answers that chat, read from the
 * server (GET /api/powered-by), so it follows the app's SPADESK_MODELS
 * setting. If it cannot be read, the line stays hidden.
 */

/**
 * Fills in and shows the #chat-powered line.
 * @param {"customer"|"owner"} chat which chat's provider to show: Pebby's, or the owner's assistant's
 */
export async function showPoweredBy(chat) {
  const line = document.querySelector("#chat-powered");
  try {
    const providers = await (await fetch("/api/powered-by")).json();
    line.textContent = `Powered by ${providers[chat]}`;
    line.hidden = false;
  } catch {
    line.hidden = true;
  }
}
