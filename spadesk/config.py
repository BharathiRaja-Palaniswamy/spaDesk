"""Settings for the whole app, in one place.

Two kinds of configuration live here:

1. Constants that say HOW the app uses the business document (which block
   holds the opening hours, which section introduces the spa). They only name
   headings; the facts themselves are always read from the document.
2. `Settings`: values that can differ per machine (API model, file paths).
   Each one can be overridden with an environment variable or a line in the
   `.env` file.

Every other module imports from here, so nothing else reads `os.environ`.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Reads the `.env` file (if there is one) into environment variables. This is
# also how ANTHROPIC_API_KEY reaches the Anthropic SDK.
load_dotenv()

# The project folder (the parent of this `spadesk/` package).
ROOT = Path(__file__).resolve().parent.parent

# ---- How the app uses the business document -------------------------------

# The bold block that lists opening hours, shown in the customer site's welcome panel.
HOURS_BLOCK = "Business Hours"
# Booking (spadesk/booking/schedule.py): the section listing the team, with the
# days each member works and their specialties.
TEAM_SECTION = "Our Team"


# ---- Per-machine settings --------------------------------------------------

# Environment variables that pick a model for one profile (spadesk/models/profiles.py).
PROFILE_PREFIX = "SPADESK_PROFILE_"
# The model sets, each a high-end and a low-end model (see spadesk/models/profiles.py):
# set name -> (the variables that name its two models, and their defaults).
MODEL_SETS = {
    "claude": {"high": ("SPADESK_CLAUDE_HIGH", "anthropic:claude-opus-5-5"),
               "low": ("SPADESK_CLAUDE_LOW", "anthropic:claude-haiku-4-5-20251001")},
    "opensource": {"high": ("SPADESK_OPENSOURCE_HIGH", "nebius:Qwen/Qwen3-235B-A22B-Instruct-2507"),
                   "low": ("SPADESK_OPENSOURCE_LOW", "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507")},
}

@dataclass(frozen=True)  # frozen: settings cannot be changed after they are loaded
class Settings:
    """Values that can differ per machine. Read once by `get_settings`."""

    data_path: Path  # the business document the customer site reads
    report_data_path: Path  # the bookings JSON the report dashboard reads
    timezone: str  # the spa's time zone, used to tell the assistant the current time
    chunk_tokens: int  # RAG: how many tokens each chunk of the document holds
    chunk_overlap: int  # RAG: how many tokens a chunk shares with the one before it
    header_tokens: int  # RAG: the most tokens of a chunk's header, used for search only; 0 for no headers
    embedding_provider: str  # RAG: which provider turns text into vectors
    vector_store: str  # RAG: which vector store keeps the chunks and their vectors
    model_cache: Path  # RAG: where the local models (minilm, cross-encoder) are downloaded to, when chosen
    vector_store_path: Path  # RAG: names a local index (tests give each its own, so in-memory stores stay apart)
    pinecone_index: str  # RAG, LangChain pipeline: the Pinecone index to use
    retrieval_top_k: int  # RAG: how many chunks are given to the model for a question
    reranker: str  # RAG: which reranker reorders the candidates, or "none"
    rerank_candidates: int  # RAG: how many chunks the first search fetches for the reranker to choose from
    calendar: str  # booking: where bookings are kept: "memory", or "google" (Google Calendar)
    slot_minutes: int  # booking: appointments start on this grid, e.g. every 30 minutes from opening
    min_notice_minutes: int  # booking: the fewest minutes ahead a booking can be made, changed or cancelled
    handoffs_path: Path  # the chat: where requests for the staff to call a customer back are appended
    rag_engine: str  # which pipeline answers both chats: "langchain" (lc/) or the reference "handbuilt" (rag/)
    # The model gateway (spadesk/models/). A model is named "provider:model", e.g.
    # "nebius:Qwen/Qwen3-235B-A22B-Instruct-2507".
    model_set: str  # which set of models the app uses: "claude" or "opensource"
    model_sets: dict[str, dict[str, str]]  # set -> {"high": "provider:model", "low": "provider:model"}
    model_fallback: str  # "provider:model" to try when a profile's own model fails; "" for none
    profile_models: dict[str, str]  # profile -> "provider:model", from SPADESK_PROFILE_<NAME> variables
    nebius_base_url: str  # the address of Nebius's OpenAI-compatible API
    nebius_api_key: str = field(repr=False)  # repr=False: never shown when settings are printed
    pinecone_api_key: str = field(repr=False)  # Pinecone's key: the vector store, and its hosted models
    google_service_account: str = field(repr=False)  # booking: the service account's key, as JSON text or a file path
    calendar_owner: str  # booking: a Google account the calendars are shared with, to see them; "" for none
    voice_secret: str = field(repr=False)  # phone line: the secret ElevenLabs sends; "" switches the line off


def get_settings() -> Settings:
    """Read every setting from the environment, falling back to its default."""
    # os.getenv(name, default) returns the environment variable if it is set,
    # otherwise the default written here.
    return Settings(
        data_path=Path(os.getenv("SPADESK_DATA", ROOT / "data" / "business_doc.md")),
        report_data_path=Path(os.getenv("SPADESK_REPORT_DATA", ROOT / "data" / "spadesk_bookings.json")),
        timezone=os.getenv("SPADESK_TIMEZONE", "America/Los_Angeles"),
        chunk_tokens=int(os.getenv("SPADESK_CHUNK_TOKENS", "150")),
        chunk_overlap=int(os.getenv("SPADESK_CHUNK_OVERLAP", "30")),
        header_tokens=int(os.getenv("SPADESK_HEADER_TOKENS", "50")),
        embedding_provider=os.getenv("SPADESK_EMBEDDING_PROVIDER", "pinecone"),
        vector_store=os.getenv("SPADESK_VECTOR_STORE", "pinecone"),
        model_cache=Path(os.getenv("SPADESK_MODEL_CACHE", ROOT / "storage" / "models")),
        vector_store_path=Path(os.getenv("SPADESK_VECTOR_STORE_PATH", ROOT / "storage" / "vectors")),
        pinecone_index=os.getenv("SPADESK_PINECONE_INDEX", "spadesk"),
        retrieval_top_k=int(os.getenv("SPADESK_RETRIEVAL_TOP_K", "3")),
        reranker=os.getenv("SPADESK_RERANKER", "pinecone"),
        rerank_candidates=int(os.getenv("SPADESK_RERANK_CANDIDATES", "20")),
        calendar=os.getenv("SPADESK_CALENDAR", "memory"),
        slot_minutes=int(os.getenv("SPADESK_SLOT_MINUTES", "30")),
        min_notice_minutes=int(os.getenv("SPADESK_MIN_NOTICE_MINUTES", "30")),
        handoffs_path=Path(os.getenv("SPADESK_HANDOFFS", ROOT / "storage" / "handoffs.jsonl")),
        rag_engine=os.getenv("SPADESK_RAG_ENGINE", "langchain"),
        model_set=os.getenv("SPADESK_MODELS", "claude"),
        model_sets={name: {tier: os.getenv(variable, default) for tier, (variable, default) in tiers.items()}
                    for name, tiers in MODEL_SETS.items()},
        model_fallback=os.getenv("SPADESK_MODEL_FALLBACK", ""),
        # SPADESK_PROFILE_CHEAP_ROUTER=nebius:... -> {"cheap-router": "nebius:..."}
        profile_models={name.removeprefix(PROFILE_PREFIX).lower().replace("_", "-"): value
                        for name, value in os.environ.items() if name.startswith(PROFILE_PREFIX) and value},
        nebius_base_url=os.getenv("SPADESK_NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
        nebius_api_key=os.getenv("NEBIUS_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        google_service_account=os.getenv("GOOGLE_SERVICE_ACCOUNT", ""),
        calendar_owner=os.getenv("SPADESK_CALENDAR_OWNER", ""),
        voice_secret=os.getenv("SPADESK_VOICE_SECRET", ""),
    )
