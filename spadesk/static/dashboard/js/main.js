/**
 * @file Entry point of the manage page. It connects the pieces:
 *
 *   1. ask the server what can be filtered        (api.js)
 *   2. build the filter row from that             (filters.js)
 *   3. on every filter change, fetch the report   (api.js)
 *   4. hand the report to each section to draw    (SECTIONS below)
 *
 * It also starts the tabs (tabs.js), loads the business profile when its tab
 * is first shown (profile.js), and starts the assistant chat (chat/widget.js).
 *
 * This is the only file index.html loads; everything else is imported from here.
 */

import { getOptions, getReport } from "./api.js";
import { initChat } from "./chat/widget.js";
import { $ } from "./dom.js";
import { initFilters } from "./filters.js";
import { showProfile } from "./profile.js";
import { dateRange } from "./format.js";
import { renderCategoryRevenue } from "./sections/category-revenue.js";
import { renderChannelRevenue } from "./sections/channel-revenue.js";
import { renderCustomerMix } from "./sections/customer-mix.js";
import { renderFeedback } from "./sections/feedback.js";
import { renderRevenueTrend } from "./sections/revenue-trend.js";
import { renderTopServices } from "./sections/top-services.js";
import { renderTrafficTrend } from "./sections/traffic-trend.js";
import { initTabs } from "./tabs.js";
import { renderTiles } from "./tiles.js";

/**
 * The report returned by GET /api/report. Its main parts:
 * `filters`, `currency`, `summary`, `previous_period`, `previous_summary`,
 * `change`, `trend`, `categories`, `services`, `channels`, `customers`,
 * `feedback`.
 * (spadesk/reports/report.py builds it and lists every part.)
 * @typedef {object} Report
 */

/**
 * Everything drawn from a report. Each entry pairs a place on the page with
 * the function that fills it. To add a chart: add a box with an id to
 * index.html, write a render function in sections/, and add one line here.
 * @type {{container: string, render: (container: Element, report: Report) => void}[]}
 */
const SECTIONS = [
  { container: "#tiles", render: renderTiles },
  { container: "#revenue-trend", render: renderRevenueTrend },
  { container: "#traffic-trend", render: renderTrafficTrend },
  { container: "#category-revenue", render: renderCategoryRevenue },
  { container: "#top-services", render: renderTopServices },
  { container: "#channel-revenue", render: renderChannelRevenue },
  { container: "#customer-mix", render: renderCustomerMix },
  { container: "#feedback", render: renderFeedback },
];

const reportArea = $("#report");
const errorLine = $("#error");
/** Counts report requests, so a slow old answer can be told from the newest one. */
let latestRequest = 0;
/** The report currently on screen, kept so it can be redrawn on resize or tab change. @type {?Report} */
let latestReport = null;

/**
 * Draws every section from one report.
 * @param {Report} report
 */
function draw(report) {
  for (const section of SECTIONS) section.render($(section.container), report);
}

/**
 * Fetches the report for a set of filters and redraws the page. Called by the
 * filter row on every change.
 * @param {{start: string, end: string, category: string, channel: string}} filters
 */
async function refresh(filters) {
  // Number each request. If the visitor changes filters quickly, answers can
  // arrive out of order; only the answer to the newest request is drawn.
  const request = ++latestRequest;
  reportArea.classList.add("loading"); // keep the old numbers, dimmed, while loading
  try {
    const report = await getReport(filters);
    if (request !== latestRequest) return;
    errorLine.hidden = true;
    $("#period").textContent = periodText(report);
    latestReport = report;
    draw(report);
  } catch (error) {
    if (request !== latestRequest) return;
    errorLine.textContent = error.message;
    errorLine.hidden = false;
  } finally {
    if (request === latestRequest) reportArea.classList.remove("loading");
  }
}

/**
 * The line under the filters that says which dates are shown.
 * @param {Report} report
 * @returns {string} e.g. "1 Sep 2026 – 30 Sep 2026 · compared with 2 Aug 2026 – 31 Aug 2026"
 */
function periodText(report) {
  const shown = dateRange(report.filters.start, report.filters.end);
  const before = report.previous_period;
  return before ? `${shown} · compared with ${dateRange(before.start, before.end)}` : shown;
}

/**
 * Runs once when the page opens: loads the filter options, fills in the
 * business name, and builds the filter row (which triggers the first report).
 */
async function start() {
  try {
    const options = await getOptions();
    document.title = `${options.business} — Manage`;
    $("#business-name").textContent = options.business;
    $("#note").textContent = options.note;
    initFilters(options, refresh);
  } catch (error) {
    errorLine.textContent = error.message;
    errorLine.hidden = false;
  }
}

// Charts are drawn at the pixel width of their card, so they are redrawn when
// the window changes size. The timer waits until the resizing has stopped.
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => latestReport && draw(latestReport), 150);
});

// What to do when a tab is shown. The profile is fetched the first time its
// tab opens. Charts drawn while the Reports tab was hidden had no width to
// measure, so they are redrawn each time that tab is shown.
initTabs((name) => {
  if (name === "profile") showProfile($("#profile"));
  if (name === "reports" && latestReport) draw(latestReport);
});

initChat();

start();
