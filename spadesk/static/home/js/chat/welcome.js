/**
 * @file What the chat shows before the visitor has asked anything: a short
 * greeting and a few questions to start from.
 *
 * This is page wording, not business data, so it is written here. Change the
 * text or the questions freely; nothing else depends on them.
 */

/**
 * The assistant's opening message.
 * @param {string} business the spa's name, from /api/spa
 * @returns {string}
 */
export const greeting = (business) =>
  `Hello 👋 I'm Pebby, your virtual assistant. Welcome to ${business}! Ask me about our `
  + "services and prices, opening hours, offers or policies. Pick a question below or type your own.";

/** Questions offered under the greeting. */
export const STARTER_QUESTIONS = [
  "What are your opening hours?",
  "Which massages do you offer?",
  "Do you have services for kids?",
  "What is your cancellation policy?",
];
