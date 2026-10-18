"""LangChain step 1: load the business document and cut it into chunks.

In LangChain, text moves through the pipeline as `Document` objects: the text
(`page_content`) plus a dict of facts about it (`metadata`). Here the
business document becomes one Document, and the splitter turns it into one
Document per chunk.

The cut is the same as the hand-built one (rag/chunking.py): fixed windows of
SPADESK_CHUNK_TOKENS tokens, each sharing SPADESK_CHUNK_OVERLAP tokens with the
one before, counted with tiktoken's "cl100k_base" encoding. LangChain's
TokenTextSplitter does exactly that. (Its RecursiveCharacterTextSplitter,
the usual default, prefers to cut between paragraphs instead; the chunks
stay fixed-size here because that is the choice made for this project.)

Each chunk's metadata says which headings its text covers ("sections"), so
answers can cite them, exactly as the hand-built chunks do.
"""

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from ..config import get_settings
from ..rag.chunking import ENCODING, _section_spans  # the same heading rules as the hand-built chunks


def load_document() -> Document:
    """The business document as one LangChain Document."""
    path = get_settings().data_path
    return Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})


def split_document(document: Document) -> list[Document]:
    """One Document per fixed-size chunk, best read in order.

    Each chunk's metadata holds:
      id        its position, e.g. "chunk-3" (the same ids as the hand-built chunks)
      start     where it begins in the document, as a character offset
      sections  the headings whose content it includes, in document order
    """
    settings = get_settings()
    splitter = TokenTextSplitter(encoding_name=ENCODING, chunk_size=settings.chunk_tokens,
                                 chunk_overlap=settings.chunk_overlap)
    text = document.page_content
    labels = _section_spans(text)  # (heading, where its content begins, where it ends)
    chunks, begin = [], 0
    for number, chunk in enumerate(splitter.split_documents([document])):
        # Where the chunk sits in the text, searching on from the previous
        # chunk's start (chunks overlap, so it begins after that, not after
        # its end). The splitter's own add_start_index option is not used: it
        # counts the overlap in characters, but this splitter overlaps by
        # tokens, so it loses its place and reports -1.
        begin = text.index(chunk.page_content, begin)
        end = begin + len(chunk.page_content)
        sections = list(dict.fromkeys(  # dict.fromkeys removes repeats, keeping order
            name for name, line_begin, line_end in labels if line_begin < end and line_end > begin))
        chunks.append(Document(page_content=chunk.page_content,
                               metadata={**chunk.metadata, "id": f"chunk-{number}", "start": begin,
                                         "sections": sections}))
    return chunks
