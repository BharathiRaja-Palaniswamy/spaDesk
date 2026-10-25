/**
 * @file The tabs at the top of the manage page ("Business profile", "Reports", "Feedback").
 *
 * The tab buttons and their panels are written in index.html. A button names
 * its tab in `data-tab`, and its panel has the id "panel-<name>". Exactly one
 * panel is visible at a time; the others carry the `hidden` attribute.
 *
 * The active tab is kept in the address as a hash (/manage#reports), so a
 * refresh or a shared link opens the same tab.
 */

/** The tab shown when the address has no (or an unknown) hash. */
const DEFAULT_TAB = "profile";

/**
 * Wires up the tab buttons and shows the starting tab.
 * @param {(name: string) => void} onShow called every time a tab becomes
 *   visible, with its name. Charts need this: they measure their width when
 *   drawn, and a hidden panel has no width.
 */
export function initTabs(onShow) {
  const buttons = [...document.querySelectorAll("[data-tab]")];
  const names = buttons.map((button) => button.dataset.tab);

  /**
   * Shows one tab's panel and hides the rest.
   * @param {string} name a `data-tab` value
   */
  function show(name) {
    for (const button of buttons) {
      const active = button.dataset.tab === name;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1; // only the active tab is in the Tab-key order
      document.getElementById(`panel-${button.dataset.tab}`).hidden = !active;
    }
    history.replaceState(null, "", `#${name}`);
    onShow(name);
  }

  /** The tab named in the address, or the default. */
  const fromAddress = () => {
    const name = location.hash.slice(1);
    return names.includes(name) ? name : DEFAULT_TAB;
  };

  buttons.forEach((button, index) => {
    button.addEventListener("click", () => show(button.dataset.tab));
    // Left and right arrow keys move between tabs, as in a desktop tab bar.
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const next = buttons[(index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length];
      next.focus();
      show(next.dataset.tab);
    });
  });

  // The back button, or a typed hash, changes the tab too.
  window.addEventListener("hashchange", () => show(fromAddress()));

  show(fromAddress());
}
