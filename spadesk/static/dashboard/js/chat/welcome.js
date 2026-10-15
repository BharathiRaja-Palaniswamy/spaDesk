/**
 * @file What the chat shows before the owner has asked anything: a short
 * introduction and a few questions to start from.
 *
 * This is page wording, not business data, so it is written here. Change the
 * text or the questions freely; nothing else depends on them.
 */

/**
 * The assistant's opening message.
 * @param {string} business the business name shown at the top of the page
 * @returns {string}
 */
export const introduction = (business) =>
  `Hi, I'm the ${business} assistant. I can answer questions about your business profile `
  + "(services and prices, opening hours, offers and policies) and about your reports "
  + "(revenue, bookings, channels and feedback). Pick a question below or type your own.";

/** Questions offered under the introduction: some answered from the business profile, some from the reports. */
export const STARTER_QUESTIONS = [
  "What are our opening hours?",
  "What is our cancellation policy?",
  "How did we do last month?",
  "Which channel brings the most bookings?",
];
