/**
 * @file The only file that fetches data from the server for the manage page.
 * (The assistant chat has its own request, in chat/assistant.js.)
 */

import { checkSignedIn } from "./session.js";

/**
 * Fetches a URL and returns its JSON body.
 * @param {string} url
 * @returns {Promise<object>} the parsed response
 * @throws {Error} with a readable message when the server answers with an error
 */
async function getJson(url) {
  const response = await fetch(url);
  checkSignedIn(response);
  const body = await response.json();
  if (!response.ok) {
    // The server sends a readable message in `detail` for filters it rejects.
    throw new Error(typeof body.detail === "string" ? body.detail : "The data could not be loaded.");
  }
  return body;
}

/**
 * Asks what the filter row can offer.
 * @returns {Promise<{business: string, note: string, min_date: string, max_date: string,
 *   categories: string[], channels: string[]}>}
 */
export const getOptions = () => getJson("/api/report/options");

/**
 * Fetches the business profile for the "Business profile" tab.
 * @returns {Promise<{name: string, sections: object[]}>} see spadesk/business/profile.py
 */
export const getProfile = () => getJson("/api/profile");

/**
 * Fetches the customers' feedback cards for the "Feedback" tab, open ones first.
 * @returns {Promise<object[]>} see spadesk/agent/feedback.py (cards)
 */
export const getFeedbackCards = () => getJson("/api/feedback");

/**
 * Closes a feedback card, with the owner's note.
 * @param {string} id the card's reference, e.g. "FB-1A2B3C"
 * @param {string} note what was done (may be empty)
 * @returns {Promise<object>} the card, now closed
 */
export async function closeFeedbackCard(id, note) {
  const response = await fetch(`/api/feedback/${encodeURIComponent(id)}/close`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ note }),
  });
  checkSignedIn(response);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The card could not be closed.");
  return body;
}

/**
 * Fetches the report for a set of filters. Empty values are left out of the
 * request, which the server reads as "no filter".
 * @param {{start: string, end: string, category: string, channel: string}} filters
 * @returns {Promise<object>} the report (see the Report type in main.js)
 */
export function getReport(filters) {
  const query = new URLSearchParams();
  for (const [name, value] of Object.entries(filters)) {
    if (value) query.set(name, value);
  }
  return getJson(`/api/report?${query}`);
}
