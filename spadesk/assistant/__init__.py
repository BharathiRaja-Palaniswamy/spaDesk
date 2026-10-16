"""The AI assistants, and the pieces they share.

There are two assistants. Both answer from the business profile; they differ
in who they talk to and what they may use:

  owner      on the manage page (route in reports/routes.py). May also call
             the report tool for revenue and booking figures.
  customer   on the customer site (customer.py). Answers from the profile
             only, so business figures can never reach a customer.

Shared pieces, one module each:

  replies.py   the question the browser sends, and one entry of a reply list
  prompt.py    the system prompt: rules for using the profile, the current
               time, and the profile itself (whole, or RAG chunks)
  model.py     the call to the model, traced in LangSmith
"""
