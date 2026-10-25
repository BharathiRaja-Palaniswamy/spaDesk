/**
 * @file The "Feedback" tab: what customers told Pebby about their visits, as
 * the Feedback agent filed it, one card each. Open cards come first; the
 * owner closes a card once it is dealt with, with a note if they like.
 *
 * Data: GET /api/feedback = [{ id, at, name, phone, words, sentiment, topic,
 *   summary, complaint, status: "open" | "closed", closed_at, owner_note }]
 * (spadesk/agent/feedback.py files and reads them).
 */

import { closeFeedbackCard, getFeedbackCards } from "./api.js";
import { el } from "./dom.js";

/** How a sentiment is shown: its word, and the class that colours it. */
const FEELINGS = {
  positive: "Happy", neutral: "Neutral", negative: "Unhappy", mixed: "Mixed",
};

/**
 * Fetches the cards and draws them. Called each time the tab is shown, so
 * new feedback appears without reloading the page.
 * @param {Element} container the #feedback-cards box
 */
export async function showFeedbackCards(container) {
  try {
    const cards = await getFeedbackCards();
    container.replaceChildren(...(cards.length ? cards.map((card) => feedbackCard(card, container))
      : [el("p", { className: "panel-empty", textContent: "No feedback yet." })]));
  } catch (error) {
    container.replaceChildren(el("p", { className: "panel-empty", textContent: error.message }));
  }
}

/**
 * One card: who and when, how they feel and about what, the summary, their own
 * words, and either a close form (open) or how it was closed.
 * @param {object} card
 * @param {Element} container redrawn after closing
 * @returns {HTMLElement}
 */
function feedbackCard(card, container) {
  const when = new Date(card.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  const feeling = card.unread ? "Not read" : FEELINGS[card.sentiment] || card.sentiment;
  return el("article", { className: `feedback-card ${card.status}${card.complaint ? " complaint" : ""}` },
    el("header", {},
      el("span", { className: `feeling ${card.unread ? "unread" : card.sentiment}`, textContent: feeling }),
      card.topic ? el("span", { className: "topic", textContent: card.topic }) : "",
      card.complaint ? el("span", { className: "flag", textContent: "Follow up" }) : "",
      el("span", { className: "card-id", textContent: card.id })),
    el("p", { className: "summary", textContent: card.summary || card.words }),
    el("blockquote", { textContent: `“${card.words}”` }),
    el("p", { className: "who", textContent: `${card.name} · ${card.phone} · ${when}` }),
    card.status === "open" ? closeForm(card, container)
      : el("p", { className: "closed-note",
        textContent: `Closed ${new Date(card.closed_at).toLocaleDateString()}${card.owner_note ? `: ${card.owner_note}` : ""}` }));
}

/**
 * The close form under an open card: an optional note and a Close button.
 * @param {object} card
 * @param {Element} container redrawn once the card is closed
 * @returns {HTMLElement}
 */
function closeForm(card, container) {
  const note = el("input", { type: "text", placeholder: "What was done (optional)", maxLength: 500 });
  const button = el("button", { type: "submit", className: "close-card", textContent: "Close" });
  return el("form", {
    className: "close-form",
    onsubmit: async (event) => {
      event.preventDefault(); // stay on the page
      button.disabled = true;
      try {
        await closeFeedbackCard(card.id, note.value);
        await showFeedbackCards(container);
      } catch (error) {
        button.disabled = false;
        button.textContent = error.message;
      }
    },
  }, note, button);
}
