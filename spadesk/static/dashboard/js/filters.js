/**
 * @file Builds the filter row and reports every change.
 */

import { $, el } from "./dom.js";
import { DEFAULT_PRESET, PRESETS } from "./presets.js";

/**
 * Fills the filter controls and wires them up.
 *
 * `onChange` is called once at the start (with the default preset applied)
 * and again whenever the visitor changes anything.
 *
 * @param {{min_date: string, max_date: string, categories: string[], channels: string[]}} options
 *   from GET /api/report/options
 * @param {(filters: {start: string, end: string, category: string, channel: string}) => void} onChange
 *   receives the current value of every filter; "" means "no filter"
 */
export function initFilters(options, onChange) {
  const start = $("#start");
  const end = $("#end");
  const category = $("#category");
  const channel = $("#channel");
  const presets = $("#presets");

  // The date inputs cannot go outside the range the data covers.
  for (const input of [start, end]) {
    input.min = options.min_date;
    input.max = options.max_date;
  }

  fillSelect(category, "All categories", options.categories);
  fillSelect(channel, "All channels", options.channels);

  /** Reads the current value of every control. */
  const current = () => ({ start: start.value, end: end.value, category: category.value, channel: channel.value });

  /**
   * Marks one preset button as selected.
   * @param {?string} presetId null for a custom range, so none is selected
   */
  function highlight(presetId) {
    for (const button of presets.children) {
      button.setAttribute("aria-pressed", String(button.dataset.preset === presetId));
    }
  }

  /**
   * Sets both dates from a preset and reports the change.
   * @param {{id: string, range: Function}} preset one entry of PRESETS
   */
  function applyPreset(preset) {
    const range = preset.range(options.min_date, options.max_date);
    start.value = range.start;
    end.value = range.end;
    highlight(preset.id);
    onChange(current());
  }

  presets.replaceChildren(...PRESETS.map((preset) => {
    const button = el("button", { type: "button", textContent: preset.label, onclick: () => applyPreset(preset) });
    button.dataset.preset = preset.id;
    return button;
  }));

  // Typing a date makes it a custom range, so no preset stays highlighted.
  for (const input of [start, end]) {
    input.addEventListener("change", () => {
      if (!start.value || !end.value) return; // wait until both dates are filled in
      highlight(null);
      onChange(current());
    });
  }
  for (const select of [category, channel]) {
    select.addEventListener("change", () => onChange(current()));
  }

  applyPreset(PRESETS.find((preset) => preset.id === DEFAULT_PRESET));
}

/**
 * Fills a dropdown. Its first entry means "no filter" and has an empty value.
 * @param {HTMLSelectElement} select
 * @param {string} allLabel text of the first entry, e.g. "All categories"
 * @param {string[]} values the other entries
 */
function fillSelect(select, allLabel, values) {
  select.replaceChildren(
    el("option", { value: "", textContent: allLabel }),
    ...values.map((value) => el("option", { value, textContent: value })),
  );
}
