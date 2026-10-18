/**
 * @file Opens and closes the chat panel. Any button with a data-open-chat
 * attribute opens it (the avatar in the corner and the "Ask Pebby!" bubble
 * beside it); the × button, the Escape key and a click on the dimmed page
 * close it.
 *
 * Opening adds the "open" class, and css/chat-panel.css slides the panel in
 * from the right. While it is open, the avatar in the corner is hidden: it
 * would sit under the panel (see css/chat-launcher.css).
 */

import { $ } from "../dom.js";

/**
 * Wires up the open and close controls of the chat panel in index.html.
 */
export function initChatWindow() {
  const panel = $("#chat-panel");
  const veil = $("#chat-veil");
  const launcher = $(".chat-launcher");
  /** The button that opened the panel, so focus can go back to it on close. */
  let opener = null;

  /**
   * Slides the panel in or out.
   * @param {boolean} open
   */
  function setOpen(open) {
    panel.classList.toggle("open", open);
    veil.classList.toggle("open", open);
    launcher.setAttribute("aria-expanded", String(open));
    if (open) {
      // Focus waits until the slide has started, so the page does not jump.
      requestAnimationFrame(() => $("#chat-input").focus());
    } else if (opener) {
      opener.focus();
    }
  }

  document.querySelectorAll("[data-open-chat]").forEach((button) => button.addEventListener("click", () => {
    opener = button.offsetParent ? button : null; // a hidden button cannot take focus back
    setOpen(true);
  }));
  $("#chat-close").addEventListener("click", () => setOpen(false));
  veil.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && panel.classList.contains("open")) setOpen(false);
  });
}
