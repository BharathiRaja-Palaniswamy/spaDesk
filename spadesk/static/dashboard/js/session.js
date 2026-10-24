/**
 * @file The owner's sign-in, as the manage page sees it.
 *
 * The page and its data are the owner's only (spadesk/access/). When the
 * sign-in runs out while the page is open, the server answers 401, and the
 * page goes to the sign-in page, which brings the owner back here after.
 */

import { $ } from "./dom.js";

/** Goes to the sign-in page if `response` says the owner is no longer signed in. */
export function checkSignedIn(response) {
  if (response.status === 401) location.assign("/login?next=/manage");
}

/** Wires up the "Sign out" button. */
export function initSignOut() {
  $("#sign-out").addEventListener("click", async () => {
    await fetch("/api/logout", { method: "POST" });
    location.assign("/login");
  });
}
