/**
 * @file Draws the business document below the hero, one band per entry in
 * bands.js, and the header menu that jumps to each band.
 *
 * Data: GET /api/profile, the whole document as sections and groups (see
 * build_profile() in spadesk/business/profile.py).
 */

import { $, el } from "../dom.js";
import { BANDS } from "./bands.js";
import { foldout, groupBlock } from "./blocks.js";
import { groupPills } from "./pills.js";

/**
 * Loads the whole business document from the server.
 * @returns {Promise<{name: string, sections: {title: string, groups: object[]}[]}>}
 */
export async function loadProfile() {
  const response = await fetch("/api/profile");
  return response.json();
}

/**
 * The bands to draw: those in bands.js whose sections exist, then one band
 * for each document section that bands.js does not mention.
 * @param {{title: string}[]} sections the document's sections
 * @returns {import("./bands.js").Band[]}
 */
function bandsFor(sections) {
  const titles = new Set(sections.map((section) => section.title));
  const listed = new Set(BANDS.flatMap((band) => band.sections));
  const known = BANDS
    .map((band) => ({ ...band, sections: band.sections.filter((title) => titles.has(title)) }))
    .filter((band) => band.sections.length);
  const extra = sections.filter((section) => !listed.has(section.title)).map((section) => ({
    // "Gift Cards" -> "gift-cards", used as the band's link target.
    id: section.title.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    label: section.title, title: section.title, sections: [section.title], layout: "cards",
  }));
  return [...known, ...extra];
}

/**
 * One document section's content: its groups, laid out by the band's layout.
 * Titled groups become pills (cards layout) or foldouts (accordion layout).
 * @param {{groups: object[]}} section
 * @param {"cards"|"accordion"} layout
 * @param {Set<string>} bookable the services the booking form offers
 * @returns {HTMLElement[]}
 */
function sectionContent(section, layout, bookable) {
  const draw = (group) => groupBlock(group, layout, bookable);
  const plain = section.groups.filter((group) => !group.title).map(draw);
  const titled = section.groups.filter((group) => group.title);
  if (!titled.length) return plain;
  if (layout === "accordion") {
    return [...plain, el("div", { className: "foldouts" }, ...titled.map((group) => foldout(group.title, [draw(group)])))];
  }
  return [...plain, groupPills(titled, draw)];
}

/**
 * Draws every band into #sections and its link into the header menu.
 * @param {{sections: {title: string, groups: object[]}[]}} profile from loadProfile()
 * @param {{name: string}[]} services the services the booking form offers, from /api/booking/options
 */
export function renderSections(profile, services) {
  const bookable = new Set(services.map((service) => service.name));
  const byTitle = new Map(profile.sections.map((section) => [section.title, section]));
  const bands = bandsFor(profile.sections);

  $("#sections").replaceChildren(...bands.map((band, index) =>
    el("section", { className: `band${index % 2 ? " band-tint" : ""}`, id: band.id },
      el("div", { className: "band-inner" },
        el("p", { className: "eyebrow", textContent: band.label }),
        el("h2", { textContent: band.title }),
        ...band.sections.flatMap((title) => [
          // A band with several sections names each one; with one, the band's title is enough.
          band.sections.length > 1 ? el("h3", { textContent: title }) : null,
          ...sectionContent(byTitle.get(title), band.layout, bookable),
        ].filter(Boolean))))));

  $("#site-nav").replaceChildren(...bands.map((band) =>
    el("a", { href: `#${band.id}`, textContent: band.label })));

  // A link like /#faq arrives before the bands exist, so the browser could not
  // scroll to it. Now that they are drawn, scroll there.
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}
