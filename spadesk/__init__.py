"""spaDesk: a spa's customer site and report dashboard. Start reading at app.py.

One thing happens as soon as the package is imported, before any connection
is opened: Python is told to trust the certificates this machine trusts.

Every HTTPS connection (to Claude, Pinecone, the model downloads) is checked
against a list of trusted certificate authorities. Python's HTTP libraries
bring their own list, which can miss one the machine trusts, for example the
certificate of a hosting provider's network filter. Connections then fail
with "certificate verify failed" although the machine itself would accept
them. truststore makes Python use the machine's own list instead, the same
one a browser or curl uses. Where the two lists agree (most machines),
nothing changes.
"""

import truststore

truststore.inject_into_ssl()
