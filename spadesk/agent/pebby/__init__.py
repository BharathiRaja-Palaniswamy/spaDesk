"""Agent 1, Pebby: the only one who talks to the customer.

    prompt.py     its instructions and the shape of its replies
    tools.py      search the guide, ask the Booking agent, record feedback, hand off
    identify.py   name and phone first
    feedback.py   reading feedback (a small model); a complaint goes to the staff
    nodes.py      the conversation's steps, and what it remembers
    graph.py      how the steps are joined, and the memory between messages
    turns.py      one turn at a time per conversation

The rules it follows are in ../rules.py.
"""
