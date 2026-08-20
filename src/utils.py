import unicodedata

def stripAccents(name):
    decompose = unicodedata.normalize("NFKD", name)
    inBytes = decompose.encode("ascii", "ignore")
    return inBytes.decode("ascii")


def normalizeName(name):
    strippedName = stripAccents(name)
    lowercase = strippedName.lower()
    return lowercase.strip()

print(f"Test: {normalizeName("José GARCÍA")}, {normalizeName("CRistianó ROnalDÓ")}")
