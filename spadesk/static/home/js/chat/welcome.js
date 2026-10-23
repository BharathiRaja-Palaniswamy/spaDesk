/**
 * @file What the chat says before the customer has written anything.
 *
 * Pebby asks for the customer's name and phone number first: the server will
 * not go on without them (spadesk/agent/identify.py), so the greeting asks
 * straight away. This is page wording, not business data, so it is written
 * here; change it freely.
 */

/**
 * The assistant's opening message.
 * @param {string} business the spa's name, from /api/spa
 * @returns {string}
 */
export const greeting = (business) =>
  `Hello 👋 I'm Pebby, your virtual assistant at ${business}. I can answer your questions and book or `
  + "cancel appointments. Before we start, may I have your name and phone number?";
