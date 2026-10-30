/**
 * @file The 👍 / 👎 under a chat reply, shared by the customer chat and the
 * owner's chat. A tap sends the reply's trace id and the rating to POST
 * /api/rating (spadesk/ratings.py), which saves it in LangSmith on the trace
 * that made the reply, so a 👎 can be found and reviewed. One tap per reply:
 * the chosen thumb stays lit and both are then disabled.
 */

/**
 * The two thumbs for one reply, or nothing when the reply has no trace (tracing off).
 * @param {string|null|undefined} traceId the reply's `trace_id`
 * @param {"customer"|"owner"} chat which chat the reply is in
 * @returns {HTMLElement|""}
 */
export function ratingButtons(traceId, chat) {
  if (!traceId) return "";
  const row = document.createElement("div");
  row.className = "chat-rating";
  const thumb = (score, symbol, label) => {
    const button = document.createElement("button");
    Object.assign(button, { type: "button", className: "chat-rating-thumb", textContent: symbol, title: label });
    button.setAttribute("aria-label", label);
    button.addEventListener("click", () => {
      row.querySelectorAll("button").forEach((each) => { each.disabled = true; });
      button.classList.add("chosen");
      // The chat goes on whether or not the rating is saved, so nothing waits on it.
      fetch("/api/rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, score, chat }),
      }).catch(() => {});
    });
    return button;
  };
  row.append(thumb(1, "👍", "Helpful"), thumb(0, "👎", "Not helpful"));
  return row;
}
