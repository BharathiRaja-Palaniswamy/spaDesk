/**
 * @file Customer site logic, in two parts:
 *
 *   1. Spa information: fetch /api/spa and fill the page with it.
 *   2. Mock booking request: post the form to /api/booking-requests.
 *
 * Everything about the spa comes from /api/spa; nothing is hard-coded here.
 */

/**
 * Finds the first element matching a CSS selector.
 * @param {string} selector
 * @returns {?Element}
 */
const $ = (selector) => document.querySelector(selector);

/**
 * Builds an element. Text is always set with textContent, never innerHTML, so
 * nothing from the server can inject HTML into the page.
 * @param {string} tag
 * @param {object} [props] properties to set, e.g. { className: "card", textContent: "hi" }
 * @param {...(Node|string|null)} children added inside, in order; null and "" are skipped
 * @returns {HTMLElement}
 */
function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((child) => child !== null && child !== ""));
  return node;
}

/* ---------- Spa information ---------- */

/**
 * Builds the card for one featured service.
 * @param {{name: string, category: string, description: string, duration: string, price: string}} service
 * @returns {HTMLElement}
 */
function serviceCard(service) {
  return el("article", { className: "card" },
    el("span", { className: "category", textContent: service.category }),
    el("h3", { textContent: service.name }),
    el("p", { textContent: service.description }),
    el("div", { className: "meta" },
      el("span", { textContent: service.duration }),
      el("span", { textContent: service.price })),
    el("div", { className: "actions" },
      el("button", { className: "link", textContent: "Request booking", onclick: () => openBooking(service.name) })));
}

/**
 * Fills every part of the page from the /api/spa response: name, hours,
 * featured services, the full price list, policy highlights, contact details
 * and the booking form's service list.
 * @param {object} spa see KnowledgeBase.public() in spadesk/business/knowledge.py
 */
function renderSpa(spa) {
  document.title = spa.name;
  document.querySelectorAll("[data-spa-name]").forEach((node) => (node.textContent = spa.name));
  $("#about").textContent = spa.about;

  $("#hero-hours").replaceChildren(...spa.hours.map((h) =>
    el("li", {}, el("span", { textContent: h.days }), el("span", { textContent: h.times }))));

  const byName = new Map(spa.services.map((s) => [s.name, s]));
  $("#featured").replaceChildren(...spa.featured.map((name) => serviceCard(byName.get(name))));

  // Full price list, grouped by category.
  const groups = new Map();
  for (const s of spa.services) groups.set(s.category, [...(groups.get(s.category) || []), s]);
  $("#menu").replaceChildren(...[...groups].map(([category, services]) =>
    el("div", { className: "menu-group" },
      el("h3", { textContent: category }),
      ...services.map((s) => el("div", { className: "menu-row" },
        el("span", {}, s.name, s.duration ? el("small", { textContent: s.duration }) : ""),
        el("span", { textContent: s.price }))))));

  $("#highlights").replaceChildren(...spa.highlights.map((h) =>
    el("article", { className: "card" },
      el("h3", { textContent: h.title }),
      h.points.length === 1
        ? el("p", { textContent: h.points[0] })
        : el("ul", {}, ...h.points.map((point) => el("li", { textContent: point }))))));

  $("#contact").replaceChildren(...Object.entries(spa.contact).flatMap(([label, value]) =>
    [el("dt", { textContent: label }), el("dd", { textContent: value })]));

  bookingForm.service.replaceChildren(...spa.services.map((s) => el("option", { textContent: s.name })));
}

/* ---------- Mock booking request ---------- */

const dialog = $("#booking");
const bookingForm = $("#booking-form");
const bookingError = $("#booking-error");

/**
 * Opens the booking request pop-up with a fresh form.
 * @param {string} [service] a service name to pre-select
 */
function openBooking(service) {
  bookingForm.hidden = false;
  $("#booking-done").hidden = true;
  bookingError.hidden = true;
  if (service) bookingForm.service.value = service;
  // Earliest selectable date is today, in the visitor's own time zone.
  const now = new Date();
  bookingForm.preferred_date.min = new Date(now - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  dialog.showModal();
}

document.querySelectorAll("[data-open-booking]").forEach((button) => button.addEventListener("click", () => openBooking()));
document.querySelectorAll("[data-close-booking]").forEach((button) => button.addEventListener("click", () => dialog.close()));

bookingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!bookingForm.reportValidity()) return;
  const submit = bookingForm.querySelector("[type=submit]");
  submit.disabled = true;
  bookingError.hidden = true;
  try {
    const response = await fetch("/api/booking-requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new FormData(bookingForm))), // form fields -> JSON
    });
    const result = await response.json();
    if (!response.ok) {
      // The server sends a readable message for its own rules, and a list for field errors.
      throw new Error(typeof result.detail === "string" ? result.detail : "Please check the details and try again.");
    }
    $("#booking-message").textContent = result.message;
    $("#booking-reference").textContent = result.reference;
    bookingForm.hidden = true;
    $("#booking-done").hidden = false;
    bookingForm.reset();
  } catch (error) {
    bookingError.textContent = error instanceof TypeError ? "We couldn't send your request. Please try again." : error.message;
    bookingError.hidden = false;
  } finally {
    submit.disabled = false;
  }
});

// Start: load the spa information and build the page.
fetch("/api/spa").then((response) => response.json()).then(renderSpa);
