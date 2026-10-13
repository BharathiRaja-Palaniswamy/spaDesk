"""RAG (retrieval-augmented generation) for the assistant.

Rather than sending the whole business document to the model with every
question, RAG sends only the parts that match the question. The steps, one
module each:

  chunking.py   cut the document into chunks of a fixed number of tokens
  embeddings/   turn each chunk into a vector that stands for its meaning
  stores/       keep the chunks and their vectors where they can be searched
  index.py      build that index from the document, and rebuild it when needed
  retriever.py  find the chunks closest in meaning to a question
  context.py    lay those chunks out for the model, in place of the whole document
"""
