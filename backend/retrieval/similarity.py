from difflib import SequenceMatcher


def similarity(query, text):
    return SequenceMatcher(None, query.lower(), text.lower()).ratio()


def rank_chunks(query, chunks):

    ranked = []

    for chunk in chunks:

        score = similarity(query, chunk["text"])

        ranked.append((score, chunk))

    ranked.sort(reverse=True, key=lambda x: x[0])

    return ranked