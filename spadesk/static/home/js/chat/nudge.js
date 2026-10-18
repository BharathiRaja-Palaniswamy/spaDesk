/**
 * @file The speech bubble beside the avatar in the corner: "👋 Any
 * questions? Ask Pebby!" (css/chat-nudge.css).
 *
 * It slides out on every page load. It goes away when the visitor opens the
 * chat or dismisses it with ×, until the next load. Its "Ask Pebby!" button
 * has a data-open-chat attribute, so js/chat/window.js opens the chat from it
 * like any other such button.
 */

import { $ } from "../dom.js";

/**
 * Shows the bubble straight away, and hides it once it has done its job.
 */
export function initChatNudge() {
  const nudge = $("#chat-nudge");

  // Added on the next frame rather than now, so the browser first draws the
  // bubble in its hidden place and then plays the slide-out.
  requestAnimationFrame(() => nudge.classList.add("show"));

  const hide = () => nudge.classList.remove("show");
  $("#chat-nudge-close").addEventListener("click", hide);
  // Opening the chat from anywhere (this bubble, the avatar, a link) ends the nudge.
  document.querySelectorAll("[data-open-chat]").forEach((button) => button.addEventListener("click", hide));
}
