/**
 * @file Builds a plain data table: the "table view" twin of a chart, so every
 * number can be read without hovering.
 */

import { el } from "./dom.js";

/**
 * @param {string[]} columns the header cells
 * @param {string[][]} rows one array of cells per row; the first cell names the row
 * @returns {HTMLTableElement}
 * @example dataTable(["Period", "Revenue"], [["1 Sep", "$640"], ["2 Sep", "$512"]])
 */
export function dataTable(columns, rows) {
  return el("table", { className: "data-table" },
    el("thead", {}, el("tr", {}, ...columns.map((name) => el("th", { textContent: name, scope: "col" })))),
    el("tbody", {}, ...rows.map((cells) =>
      // The first cell names the row; the rest are its values.
      el("tr", {}, ...cells.map((cell, index) =>
        el(index === 0 ? "th" : "td", { textContent: cell, ...(index === 0 ? { scope: "row" } : {}) }))))));
}
