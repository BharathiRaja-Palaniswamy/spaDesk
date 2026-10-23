"""The phone line: Pebby answering calls through ElevenLabs.

    call.py    which call a request belongs to, who is calling, what they said
    stream.py  replies in the streamed OpenAI format ElevenLabs reads
    routes.py  POST /v1/chat/completions, protected by SPADESK_VOICE_SECRET

ElevenLabs listens, keeps the turns and speaks; Pebby (spadesk/agent/)
thinks, with the same rules, tools and calendar as the website chat.
"""
