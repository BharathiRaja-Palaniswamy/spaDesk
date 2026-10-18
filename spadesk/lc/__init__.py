"""The RAG pipeline rebuilt with LangChain, next to the hand-built one in rag/.

Both do the same job, step for step, so they can be compared with the same
evals (evals/). The hand-built version shows what each step does; this one
shows how LangChain packages the same steps:

    hand-built (rag/, assistant/)       LangChain (lc/)
    business_doc.md read as text   ->   documents.py: a Document with metadata
    chunking.py (150/30 tokens)    ->   documents.py: TokenTextSplitter, same sizes
    embeddings/ (MiniLM)           ->   embeddings.py: an Embeddings class, same model
    stores/ (Chroma)               ->   vectorstores/: any LangChain VectorStore (Chroma, Pinecone)
    retriever.py + rerankers/      ->   retriever.py + rerankers.py: a BaseRetriever with a
                                        pluggable reranking compressor
    assistant/prompt.py, model.py  ->   prompt.py + chat.py: ChatPromptTemplate + a pluggable chat
                                        model (ChatAnthropic), replying in the same structured shape
    owner tool loop                ->   (later) a LangGraph graph with the get_report tool

SPADESK_RAG_ENGINE=langchain makes the customer chat (POST /api/chat) answer
through this pipeline; "handbuilt", the default, keeps the original. The owner
chat still uses the hand-built pipeline until its LangGraph version is built.

Every provider is pluggable, as in the hand-built pipeline: the embedding
model, the vector store, the reranker and the chat model are each chosen by a
setting, behind LangChain's common interface for that kind of component
(Embeddings, VectorStore, BaseDocumentCompressor, BaseChatModel). Each
package's __init__.py lists its providers and how to add one.

Only langchain-core and the standalone packages (langchain-text-splitters,
langchain-chroma, langchain-anthropic, langgraph) are used: langchain-community
is being retired, so nothing here depends on it.
"""
