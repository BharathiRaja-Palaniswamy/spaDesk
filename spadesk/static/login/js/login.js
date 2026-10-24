/**
 * @file The owner's sign-in: sends the password to POST /api/login. On
 * success the server sets the sign-in cookie, and the page goes back to
 * where the owner was sent from (?next=..., the manage page by default).
 */

const form = document.querySelector("#sign-in-form");
const error = document.querySelector("#sign-in-error");

/**
 * Where to go after signing in: the `next` address, if it is a page of this
 * site, so a link cannot send the owner to another website after sign-in.
 * @returns {string}
 */
function nextPage() {
  const next = new URLSearchParams(location.search).get("next") || "/manage";
  return next.startsWith("/") && !next.startsWith("//") ? next : "/manage";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  error.hidden = true;
  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: form.password.value }),
    });
    if (!response.ok) throw new Error((await response.json()).detail);
    location.assign(nextPage());
  } catch (problem) {
    // A TypeError here means the request never reached the server.
    error.textContent = problem instanceof TypeError ? "We couldn't reach the server. Please try again."
      : problem.message;
    error.hidden = false;
    button.disabled = false;
  }
});
