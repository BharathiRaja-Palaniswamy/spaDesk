/**
 * @file The phone-line test: a call to Pebby from the owner's browser.
 *
 * It is ElevenLabs' own widget (the <elevenlabs-convai> element and its
 * script). It connects the browser's microphone and speaker to the same
 * ElevenLabs agent that answers the spa's phone number, and that agent asks
 * spaDesk's Pebby what to say (spadesk/voice/). So a test booking made here
 * follows the same rules, and lands on the same calendar, as a real call.
 *
 * It is on the manage page only, as a testing tool: customers book through
 * Pebby's chat on the customer site, or by phone. It is shown only when the
 * server names an agent (SPADESK_VOICE_AGENT_ID, from GET /api/voice-widget).
 */

import { el } from "./dom.js";

const SCRIPT = "https://unpkg.com/@elevenlabs/convai-widget-embed";

/** Adds the widget to the page, if an agent is set. */
export async function initVoiceWidget() {
  try {
    const { agent_id: agentId } = await (await fetch("/api/voice-widget")).json();
    if (!agentId) return;
    // "agent-id" is an attribute with a dash, so it is set as one rather than as a property.
    const widget = el("elevenlabs-convai");
    widget.setAttribute("agent-id", agentId);
    document.body.append(widget);
    document.head.append(el("script", { src: SCRIPT, async: true, type: "text/javascript" }));
  } catch {
    // No widget if the server cannot be asked; the rest of the page works as before.
  }
}
