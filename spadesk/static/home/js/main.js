/**
 * @file Starts the customer site: loads the spa's facts and the whole
 * business document, fills the page, wires up the booking pop-up and the
 * chat panel, and starts the chat.
 */

import { initBooking } from "./booking.js";
import { initChat } from "./chat/chat.js";
import { initChatWindow } from "./chat/window.js";
import { loadProfile, renderSections } from "./sections/sections.js";
import { loadSpa, renderSpa } from "./spa.js";

// The two requests do not depend on each other, so they are sent together.
const [spa, profile] = await Promise.all([loadSpa(), loadProfile()]);
renderSpa(spa);
renderSections(profile, spa.services);
initBooking(spa.services);
initChatWindow();
initChat(spa.name);
