/**
 * @file Starts the customer site: loads the spa's facts and the whole
 * business document, fills the page, wires up the booking pop-up, the
 * chat panel and the "Ask Pebby!" bubble, and starts the chat.
 */

import { initBooking, loadBookingOptions } from "./booking.js";
import { initChat } from "./chat/chat.js";
import { initChatNudge } from "./chat/nudge.js";
import { initChatWindow } from "./chat/window.js";
import { loadProfile, renderSections } from "./sections/sections.js";
import { loadSpa, renderSpa } from "./spa.js";

// The three requests do not depend on each other, so they are sent together.
const [spa, profile, bookable] = await Promise.all([loadSpa(), loadProfile(), loadBookingOptions()]);
renderSpa(spa);
renderSections(profile, bookable);
initBooking(bookable);
initChatWindow();
initChatNudge();
initChat(spa.name);
