/**
 * @file The booking pop-up: the customer's details, a service, a free time,
 * then the confirmed booking with its reference.
 *
 * Every choice comes from the server, which applies the booking rules
 * (spadesk/booking/): /api/booking/options lists what can be booked and who
 * offers it, /api/booking/free the free times for a service on a day, and
 * POST /api/bookings books one. The page never decides what is free.
 */

import { $, el } from "./dom.js";

const dialog = $("#booking");
const form = $("#booking-form");
const error = $("#booking-error");
const times = $("#booking-times");

/** @type {Map<string, {name: string, category: string, minutes: number[], prices: number[], therapists: string[]}>} */
let services = new Map();

/** Shows `message` under the form, or hides the message when it is empty. */
function showError(message) {
  error.textContent = message;
  error.hidden = !message;
}

/**
 * Reads a JSON reply, turning a refusal into an Error with the server's message.
 * @param {Response} response
 */
async function read(response) {
  const result = await response.json();
  if (!response.ok) {
    // The server sends a readable message for its own rules, and a list for field errors.
    throw new Error(typeof result.detail === "string" ? result.detail : "Please check the details and try again.");
  }
  return result;
}

/** The chosen service's lengths and therapists, in the two smaller lists. */
function fillChoices() {
  const service = services.get(form.service.value);
  form.minutes.replaceChildren(...service.minutes.map((minutes, i) =>
    el("option", { value: minutes, textContent: `${minutes} min · $${service.prices[i]}` })));
  // A service with one length needs no choice.
  $("#booking-length").hidden = service.minutes.length < 2;
  form.therapist.replaceChildren(el("option", { value: "", textContent: "Anyone" }),
    ...service.therapists.map((name) => el("option", { value: name, textContent: name })));
}

/**
 * One therapist's free times, as a row of choices. Each choice's value is
 * "therapist|HH:MM", so the form knows both when one is picked.
 */
function therapistTimes(therapist, starts) {
  return el("div", { className: "times-row" },
    el("p", { className: "times-who", textContent: therapist }),
    el("div", { className: "times-chips" }, ...starts.map((start) =>
      el("label", { className: "time" },
        el("input", { type: "radio", name: "slot", value: `${therapist}|${start}`, required: true }),
        el("span", { textContent: start })))));
}

/** Shows `text` in place of the free times. */
function timesHint(text) {
  times.replaceChildren(el("legend", { textContent: "Free times" }), el("p", { className: "hint", textContent: text }));
}

/** Asks the server for the free times of the chosen service on the chosen day. */
async function loadTimes() {
  showError("");
  if (!form.day.value) return timesHint("Choose a date to see the free times.");
  timesHint("Looking for free times…");
  const query = new URLSearchParams({ service: form.service.value, day: form.day.value, minutes: form.minutes.value });
  if (form.therapist.value) query.set("therapist", form.therapist.value);
  try {
    const { free } = await read(await fetch(`/api/booking/free?${query}`));
    const rows = Object.entries(free).map(([therapist, starts]) => therapistTimes(therapist, starts));
    if (!rows.length) return timesHint("No free times that day. Please try another date.");
    times.replaceChildren(el("legend", { textContent: "Free times" }), ...rows);
  } catch (problem) {
    timesHint("");
    showError(problem instanceof TypeError ? "We couldn't load the free times. Please try again." : problem.message);
  }
}

/**
 * Opens the pop-up with a fresh form.
 * @param {string} [service] a service name to pre-select
 */
export function openBooking(service) {
  form.hidden = false;
  $("#booking-done").hidden = true;
  showError("");
  if (service) form.service.value = service;
  fillChoices();
  // The earliest date that can be picked is today, in the visitor's own time zone.
  const now = new Date();
  form.day.min = new Date(now - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  loadTimes();
  dialog.showModal();
}

/** "2026-10-30T15:00" -> "Friday 30 October, 15:00", as written; the times are the spa's. */
function when(start, end) {
  const [day, from] = start.split("T");
  const date = new Date(`${day}T12:00`).toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
  return `${date}, ${from}–${end.split("T")[1]}`;
}

/**
 * Books the chosen time. Shows the confirmed booking on success; on a
 * refusal (e.g. the time was just taken), the message and fresh times.
 * @param {SubmitEvent} event
 */
async function send(event) {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const chosen = form.querySelector("input[name=slot]:checked");
  if (!chosen) return showError("Please choose a time.");
  const [therapist, start] = chosen.value.split("|");
  const submit = form.querySelector("[type=submit]");
  submit.disabled = true;
  showError("");
  try {
    const booked = await read(await fetch("/api/bookings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ service: form.service.value, therapist, start: `${form.day.value}T${start}`,
                             minutes: Number(form.minutes.value), name: form.name.value, phone: form.phone.value }),
    }));
    $("#booking-summary").textContent = `${booked.service} with ${booked.therapist}, ${when(booked.start, booked.end)}.`;
    $("#booking-reference").textContent = booked.reference;
    form.hidden = true;
    $("#booking-done").hidden = false;
  } catch (problem) {
    // A TypeError here means the request never reached the server.
    showError(problem instanceof TypeError ? "We couldn't book that time. Please try again." : problem.message);
    if (!(problem instanceof TypeError)) loadTimes(); // the times may have changed since they were shown
  } finally {
    submit.disabled = false;
  }
}

/**
 * Wires up the pop-up: the services, every open and close button, and the form.
 * @param {{name: string, category: string, minutes: number[], prices: number[], therapists: string[]}[]} options
 *   the bookable services, from /api/booking/options
 */
export function initBooking(options) {
  services = new Map(options.map((service) => [service.name, service]));
  // Services grouped by category, as on the page.
  const categories = new Map();
  for (const service of options) categories.set(service.category, [...(categories.get(service.category) ?? []), service]);
  form.service.replaceChildren(...[...categories].map(([category, list]) =>
    el("optgroup", { label: category }, ...list.map((s) => el("option", { textContent: s.name })))));
  form.service.addEventListener("change", () => { fillChoices(); loadTimes(); });
  for (const field of [form.minutes, form.therapist, form.day]) field.addEventListener("change", loadTimes);
  document.querySelectorAll("[data-open-booking]").forEach((button) => button.addEventListener("click", () => openBooking()));
  document.querySelectorAll("[data-close-booking]").forEach((button) => button.addEventListener("click", () => dialog.close()));
  form.addEventListener("submit", send);
}

/** The bookable services, for the pop-up and the "Book" links on the page. */
export async function loadBookingOptions() {
  return (await read(await fetch("/api/booking/options"))).services;
}
