/**
 * @file Which sections of the business document go in which band of the
 * page, in what order, and what each band is called in the header menu.
 *
 * Section names are the document's own headings. A section listed here but
 * missing from the document is skipped; a section in the document but not
 * listed here still appears, in a band of its own at the end, so nothing in
 * the document is ever left off the page.
 *
 * `layout` picks how text entries (an FAQ, an offer) are shown: as cards
 * side by side, or as questions that open on click. Priced entries are
 * always cards, whatever the layout.
 */

/** @typedef {{id: string, label: string, title: string, sections: string[], layout: "cards"|"accordion"}} Band */

/** @type {Band[]} */
export const BANDS = [
  { id: "services", label: "Services", title: "Our services", sections: ["Core Services"], layout: "cards" },
  { id: "yoga", label: "Yoga", title: "Yoga and movement", sections: ["Yoga and Movement"], layout: "cards" },
  { id: "packages", label: "Packages", title: "Packages and memberships", sections: ["Packages and Memberships"], layout: "cards" },
  { id: "offers", label: "Offers", title: "Offers and extras",
    sections: ["First-Time Customer Offer", "Seasonal Promotions", "Add-Ons"], layout: "cards" },
  { id: "about", label: "About", title: "About us", sections: ["About", "Mission", "Our Team"], layout: "cards" },
  { id: "faq", label: "FAQ", title: "Questions people ask", sections: ["Frequently Asked Questions"], layout: "accordion" },
  { id: "policies", label: "Policies", title: "Policies and suitability",
    sections: ["Policies", "Service Suitability"], layout: "accordion" },
  { id: "visit", label: "Visit", title: "Visit us", sections: ["Location and Hours", "Facilities"], layout: "cards" },
];
