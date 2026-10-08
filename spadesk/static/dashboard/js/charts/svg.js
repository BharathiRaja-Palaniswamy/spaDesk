/**
 * @file Builds SVG elements. SVG needs its own "namespace", so
 * document.createElement does not work for it; this helper hides that detail.
 */

const NAMESPACE = "http://www.w3.org/2000/svg";

/**
 * Builds one SVG element.
 * @param {string} tag e.g. "line", "rect", "path", "text"
 * @param {Object<string, string|number>} [attributes] e.g. { x1: 0, y1: 0, class: "grid-line" }
 * @param {...Node} children added inside, in order
 * @returns {SVGElement}
 * @example svg("line", { x1: 0, y1: 0, x2: 10, y2: 10, class: "grid-line" })
 */
export function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS(NAMESPACE, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  node.append(...children);
  return node;
}
