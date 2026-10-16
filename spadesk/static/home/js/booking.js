/**
 * @file The booking request pop-up: open it, send the form to
 * /api/booking-requests, and show the reference that comes back.
 *
 * It is a mock: a request is saved on the server as "requested", never as a
 * confirmed appointment.
 */

import { $, el } from "./dom.js";

const dialog = $("#booking");
const form = $("#booking-form");
const error = $("#booking-error");

/**
 * Opens the pop-up with a fresh form.
 * @param {string} [service] a service name to pre-select
 */
export function openBooking(service) {
  form.hidden = false;
  $("#booking-done").hidden = true;
  error.hidden = true;
  if (service) form.service.value = service;
  // The earliest date that can be picked is today, in the visitor's own time zone.
  const now = new Date();
  form.preferred_date.min = new Date(now - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  dialog.showModal();
}

/**
 * Sends the form. Shows the "request received" view on success, or a
 * readable message on failure.
 * @param {SubmitEvent} event
 */
async function send(event) {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const submit = form.querySelector("[type=submit]");
  submit.disabled = true;
  error.hidden = true;
  try {
    const response = await fetch("/api/booking-requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new FormData(form))), // form fields -> JSON
    });
    const result = await response.json();
    if (!response.ok) {
      // The server sends a readable message for its own rules, and a list for field errors.
      throw new Error(typeof result.detail === "string" ? result.detail : "Please check the details and try again.");
    }
    $("#booking-message").textContent = result.message;
    $("#booking-reference").textContent = result.reference;
    form.hidden = true;
    $("#booking-done").hidden = false;
    form.reset();
  } catch (problem) {
    // A TypeError here means the request never reached the server.
    error.textContent = problem instanceof TypeError ? "We couldn't send your request. Please try again." : problem.message;
    error.hidden = false;
  } finally {
    submit.disabled = false;
  }
}

/**
 * Wires up the pop-up: the service list, every open and close button, and
 * the form.
 * @param {{name: string}[]} services the spa's services, from /api/spa
 */
export function initBooking(services) {
  form.service.replaceChildren(...services.map((s) => el("option", { textContent: s.name })));
  document.querySelectorAll("[data-open-booking]").forEach((button) => button.addEventListener("click", () => openBooking()));
  document.querySelectorAll("[data-close-booking]").forEach((button) => button.addEventListener("click", () => dialog.close()));
  form.addEventListener("submit", send);
}
