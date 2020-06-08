import re

import tensorflow as tf
import tensorflow_datasets as tfds

from bs4 import BeautifulSoup

Tokenizer = tfds.features.text.Tokenizer


def cleanNonstandardSlashes(s):
    slashes = [
        u'\u2044',  # fraction slash (⁄)
        u'\u2215',  # division slash (∕)
        u'\uFF0F',  # fullwidth solidus (／)
        u'\u29F8',  # big solidus (⧸)
    ]

    for slash in slashes:
        s = s.replace(slash, '/')

    return s


def cleanUnicodeFractions(s):
    fractions = {
        u'\u215b': '1/8',
        u'\u215c': '3/8',
        u'\u215d': '5/8',
        u'\u215e': '7/8',
        u'\u2159': '1/6',
        u'\u215a': '5/6',
        u'\u2155': '1/5',
        u'\u2156': '2/5',
        u'\u2157': '3/5',
        u'\u2158': '4/5',
        u'\u00bc': '1/4',
        u'\u00be': '3/4',
        u'\u2153': '1/3',
        u'\u2154': '2/3',
        u'\u00bd': '1/2',
    }

    for f_unicode, f_ascii in fractions.items():
        s = s.replace(f_unicode, u' ' + f_ascii)

    return s


def cleanCommonEscapeSequences(s):
    escape_sequences = [
        '\\n',
        '\\t',
    ]

    for escape_sequence in escape_sequences:
        s = s.replace(escape_sequence, ' ')

    return s


def cleanHTMLTags(s):
    soup = BeautifulSoup(s, "html.parser")

    return soup.get_text()


def cleanUnitAbbreviations(s):
    s = re.sub(r'(\d+)g', r'\1 grams', s)
    s = re.sub(r'(\d+)oz', r'\1 ounces', s)

    return s


def clean(s):
    # remove any stray html tags (e.g. anchor tags)
    s = cleanHTMLTags(s)

    # replace all unicode fractions with ascii representations
    # e.g. ½ = 1/2
    s = cleanUnicodeFractions(s)

    # clean up any non-standard forward slash uses
    s = cleanNonstandardSlashes(s)

    # remove any escape sequences (e.g. \n, \t)
    s = cleanCommonEscapeSequences(s)

    # handle special unit abbreviation that are actually two words
    # e.g. 100g => 100 grams
    s = cleanUnitAbbreviations(s)

    return s


# punctuation to split on
SPLITTING_PUNCTUATION = [
    ',',
    '(',
    ')',
    '/',
]

# all allowed punctuation
ALLOWED_PUNCTUATION = [
    *SPLITTING_PUNCTUATION,
    '.',  # decimal quantities
    '-',  # e.g. all-purpose flour
]

# regex to identify all banned characters
BANNED_REGEX = rf'[^\w{re.escape("".join(ALLOWED_PUNCTUATION))}]'
SPLIT_REGEX = rf'\s|([{re.escape("".join(SPLITTING_PUNCTUATION))})])'


class IngredientPhraseTokenizer(Tokenizer):
    def tokenize(self, s):
        s = tf.compat.as_text(s)

        s = clean(s)

        # remove all non-word characters except a few approved types of punctuation
        # and replace with a space (so we split on those characters)
        s = re.sub(BANNED_REGEX, ' ', s, flags=re.ASCII)

        # split into tokens on specific approved punctuations and white space
        tokens = re.split(SPLIT_REGEX, s)

        # Filter out empty strings
        tokens = [t for t in tokens if t]

        return tokens
