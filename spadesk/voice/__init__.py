"""The phone line: Pebby answering calls through ElevenLabs.

    call.py    which call a request belongs to, who is calling, what they said
    stream.py  replies in the streamed OpenAI format ElevenLabs reads
    routes.py  POST /v1/chat/completions, protected by SPADESK_VOICE_SECRET;
               GET /api/voice-widget, for the manage page's phone-line test

Calls come in two ways, to the same ElevenLabs agent: a phone call to the
spa's Twilio number, and, for testing, the voice widget on the manage page
(the browser's microphone, no phone needed; static/dashboard/js/voice.js).
Customers only see Pebby's chat on the customer site.

ElevenLabs listens, keeps the turns and speaks; Pebby (spadesk/agent/)
thinks, with the same rules, tools and calendar as the website chat.
"""
