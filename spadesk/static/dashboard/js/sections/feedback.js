/**
 * @file The "How customers feel" card: one face for each star rating, with
 * how many visits were given that rating, and under them the feedback tags
 * customers chose, grouped by sentiment.
 *
 * Data: report.feedback = {
 *   rated, average_rating,
 *   ratings: [{ rating: 5..1, count }],
 *   sentiments: [{ sentiment, count }],
 *   tags: [{ tag, sentiment, count }]      most common first
 * }
 */

import { chartCard } from "../card.js";
import { el } from "../dom.js";
import { number, percent, rating } from "../format.js";

/** The face and the words shown for each star rating. */
const FACES = {
  5: { face: "😄", label: "Loved it" },
  4: { face: "🙂", label: "Happy" },
  3: { face: "😐", label: "It was okay" },
  2: { face: "🙁", label: "Unhappy" },
  1: { face: "😞", label: "Very unhappy" },
};

/**
 * The CSS class that colours a sentiment (see css/charts.css).
 * @param {string} sentiment "Happy", "Neutral" or "Unhappy"
 * @returns {string} e.g. "sentiment-happy"
 */
const colourClass = (sentiment) => `sentiment-${sentiment.toLowerCase()}`;

/**
 * Fills the card. Called by main.js on every filter change.
 * @param {Element} container the card's box from index.html
 * @param {object} report the /api/report response; this card uses `feedback`
 */
export function renderFeedback(container, report) {
  const { rated, average_rating, ratings, sentiments, tags } = report.feedback;
  const tagsFor = (sentiment) => tags.filter((t) => t.sentiment === sentiment);

  chartCard(container, {
    title: "How customers feel",
    subtitle: `${number(rated)} rated visits, average rating ${rating(average_rating)} out of 5`,
    empty: rated === 0,
    draw: (body) => body.replaceChildren(
      // One tile per star rating, best first.
      el("div", { className: "faces" }, ...ratings.map((row) =>
        el("div", { className: "face" },
          el("span", { className: "face-icon", textContent: FACES[row.rating].face, role: "img",
            ariaLabel: `${row.rating} out of 5` }),
          el("span", { className: "face-count", textContent: number(row.count) }),
          el("span", { className: "face-label", textContent: FACES[row.rating].label }),
          el("span", { className: "face-share", textContent: `${row.rating}★ · ${percent(row.count / rated)}` })))),
      // What they said: the tags behind each sentiment.
      el("div", { className: "feedback-breakdown" }, ...sentiments.map((s) =>
        el("section", {},
          el("h3", {},
            el("span", { className: `swatch ${colourClass(s.sentiment)}` }),
            s.sentiment,
            el("span", { className: "feedback-count", textContent: `${number(s.count)} · ${percent(s.count / rated)}` })),
          el("ul", { className: "tag-list" }, ...tagsFor(s.sentiment).map((t) =>
            el("li", {}, t.tag, el("span", { textContent: number(t.count) })))))))),
    table: {
      columns: ["Rating", "Feeling", "Visits", "Share"],
      rows: ratings.map((row) => [`${row.rating} out of 5`, FACES[row.rating].label, number(row.count), percent(row.count / rated)]),
    },
  });
}
