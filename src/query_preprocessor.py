import re
from spellchecker import SpellChecker


spell = SpellChecker()


def correct_spelling(query, document_vocabulary=None):

    if document_vocabulary is None:
        document_vocabulary = set()

    document_vocabulary = {
        word.lower() for word in document_vocabulary
    }

    words = re.findall(r"\b[\w']+\b|[^\w\s]", query)

    corrected_words = []

    for word in words:

        lower_word = word.lower()

        if (
            not word.isalpha()
            or len(word) <= 2
            or lower_word in document_vocabulary
        ):
            corrected_words.append(word)
            continue

        correction = spell.correction(lower_word)

        if correction is None:
            corrected_words.append(word)
            continue

        if word[0].isupper():
            correction = correction.capitalize()

        corrected_words.append(correction)

    corrected_query = " ".join(corrected_words)

    corrected_query = re.sub(
        r"\s+([,.?!])",
        r"\1",
        corrected_query
    )

    return corrected_query