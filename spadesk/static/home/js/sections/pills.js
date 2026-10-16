/**
 * @file A row of pills that switches between the groups of one section,
 * e.g. Facials, Massage Therapy and Kids Services under Core Services. One
 * group shows at a time, so a long list stays short on screen.
 */

import { el } from "../dom.js";

/**
 * Builds the pills and the area they switch.
 * @param {{title: string}[]} groups the section's groups; each title becomes a pill
 * @param {(group: object) => HTMLElement} draw builds the content of one group
 * @returns {HTMLElement}
 */
export function groupPills(groups, draw) {
  const area = el("div", { className: "pill-area" });
  const pills = groups.map((group) => el("button", {
    type: "button", className: "pill-button", textContent: group.title, onclick: () => show(group),
  }));

  /**
   * Shows one group and marks its pill as the pressed one.
   * @param {{title: string}} group
   */
  function show(group) {
    pills.forEach((pill) => pill.setAttribute("aria-pressed", String(pill.textContent === group.title)));
    area.replaceChildren(draw(group));
  }

  show(groups[0]);
  return el("div", {}, el("div", { className: "pill-row", role: "group", ariaLabel: "Categories" }, ...pills), area);
}
