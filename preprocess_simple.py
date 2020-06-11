'''
Preprocess NYT data using a simplier tokenization method than the one used by the
ingredient-phrase-tagger repo. A large portion of this code is borrowed from that repo,
with minor modification - the repo can be found here:
https://github.com/nytimes/ingredient-phrase-tagger
'''
import re
import decimal
import math
import os
import json

import tensorflow as tf
import tensorflow_datasets as tfds

TokenTextEncoder = tfds.features.text.TokenTextEncoder
Tokenizer = tfds.features.text.Tokenizer

import pandas as pd
from tokenizer import IngredientPhraseTokenizer, clean

ingredientPhraseTokenizer = IngredientPhraseTokenizer()


def singularize(word):
    """
    A poor replacement for the pattern.en singularize function, but ok for now.
    """

    units = {
        "cups": u"cup",
        "tablespoons": u"tablespoon",
        "teaspoons": u"teaspoon",
        "pounds": u"pound",
        "ounces": u"ounce",
        "cloves": u"clove",
        "sprigs": u"sprig",
        "pinches": u"pinch",
        "bunches": u"bunch",
        "slices": u"slice",
        "grams": u"gram",
        "heads": u"head",
        "quarts": u"quart",
        "stalks": u"stalk",
        "pints": u"pint",
        "pieces": u"piece",
        "sticks": u"stick",
        "dashes": u"dash",
        "fillets": u"fillet",
        "cans": u"can",
        "ears": u"ear",
        "packages": u"package",
        "strips": u"strip",
        "bulbs": u"bulb",
        "bottles": u"bottle"
    }

    if word in units.keys():
        return units[word]
    else:
        return word


def normalizeToken(s):
    """
    ToDo: FIX THIS. We used to use the pattern.en package to singularize words, but
    in the name of simple deployments, we took it out. We should fix this at some
    point.
    """
    return singularize(s)


def clumpFractions(s):
    """
    Replaces the whitespace between the integer and fractional part of a quantity
    with a dollar sign, so it's interpreted as a single token. The rest of the
    string is left alone.

        clumpFractions("aaa 1 2/3 bbb")
        # => "aaa 1$2/3 bbb"
    """
    return re.sub(r'(\d+)\s+(\d)/(\d)', r'\1$\2/\3', s)


def unclump(s):
    """
    Replacess $'s with spaces. The reverse of clumpFractions.
    """
    return re.sub(r'\$', " ", s)


def constructRegexForNumberTokens(numberTokens):
    regex = r''

    prevNumberToken = None
    for index, numberToken in enumerate(numberTokens):
        if index == 0:  #or numberToken == "/" or prevNumberToken == "/":
            regex += numberToken
        elif numberToken == "/" or prevNumberToken == "/":
            regex += "\s*"
            regex += numberToken
        else:
            regex += "\s+"
            regex += numberToken

        prevNumberToken = numberToken

    return regex


def matchNumberTokensToOriginalString(numberTokenGroup, original_input):
    clean_input = clean(original_input)
    match = re.search(constructRegexForNumberTokens(numberTokenGroup),
                      clean_input)

    if match:
        original_text = match.group()
    else:
        print('number tagging broke')
        print(numberTokenGroup)
        print(original_input)
        print(constructRegexForNumberTokens(numberTokenGroup))
        original_text = None

    return original_text


def row_to_example(row):
    """Translates a row of labeled data into example format.

    Args:
        row: A row of data from the input CSV of labeled ingredient data.

    Returns:
        [list of tokens, list of tags]
    """
    original_input = row['input']

    # extract original ingredient phrase
    tokens = ingredientPhraseTokenizer.tokenize(original_input)

    # grab label info
    labels = row_to_labels(row)

    # attempt to tag tokens
    tagged_tokens = []
    numberTokenGroup = []
    for token in tokens:
        if bool(re.match('^[0-9/]+$', token)):
            numberTokenGroup.append(token)
        else:
            if len(numberTokenGroup):
                if numberTokenGroup[0].isdigit():
                    original_text = matchNumberTokensToOriginalString(
                        numberTokenGroup, original_input)
                    if original_text == None:
                        print('number tagging broke (continued)')
                        print(original_input)
                        print(tokens)
                        grouped_token = "".join(numberTokenGroup)
                    else:
                        grouped_token = clumpFractions(original_text)

                    possible_tags = match_up(grouped_token, labels)

                    for numberToken in numberTokenGroup:
                        tagged_tokens.append((numberToken, possible_tags))
                else:
                    for numberToken in numberTokenGroup:
                        possible_tags = match_up(numberToken, labels)
                        tagged_tokens.append((numberToken, possible_tags))

            numberTokenGroup = []

            possible_tags = match_up(token, labels)

            tagged_tokens.append((token, possible_tags))

    if len(numberTokenGroup):
        if numberTokenGroup[0].isdigit():
            original_text = matchNumberTokensToOriginalString(
                numberTokenGroup, original_input)
            if original_text == None:
                print('number tagging broke (continued)')
                print(original_input)
                print(tokens)
                grouped_token = "".join(numberTokenGroup)
            else:
                grouped_token = clumpFractions(original_text)

            possible_tags = match_up(grouped_token, labels)

            for numberToken in numberTokenGroup:
                tagged_tokens.append((numberToken, possible_tags))
        else:
            for numberToken in numberTokenGroup:
                possible_tags = match_up(numberToken, labels)
                tagged_tokens.append((numberToken, possible_tags))

    tagged_tokens = add_prefixes(tagged_tokens)

    # select best label for each token
    example = [original_input, []]
    for token, tags in tagged_tokens:
        example[1].append(best_tag(tags))

    return example


def row_to_labels(row):
    """Extracts labels from a labelled ingredient data row.

    Args:
        A row of full data about an ingredient, including input and labels.

    Returns:
        A dictionary of the label data extracted from the row.
    """
    labels = {}
    label_keys = ['name', 'qty', 'range_end', 'unit', 'comment']
    for key in label_keys:
        labels[key] = row[key]
    return labels


def parse_numbers(s):
    """
    Parses a string that represents a number into a decimal data type so that
    we can match the quantity field in the db with the quantity that appears
    in the display name. Rounds the result to 2 places.
    """
    ss = unclump(s)

    m3 = re.match('^\d+$', ss)
    if m3 is not None:
        return decimal.Decimal(round(float(ss), 2))

    m1 = re.match(r'(\d+)\s+(\d)/(\d)', ss)
    if m1 is not None:
        num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
        return decimal.Decimal(str(round(num, 2)))

    m2 = re.match(r'^(\d)/(\d)$', ss)
    if m2 is not None:
        num = float(m2.group(1)) / float(m2.group(2))
        return decimal.Decimal(str(round(num, 2)))

    return None


def match_up(token, labels):
    """
    Returns our best guess of the match between the tags and the
    words from the display text.

    This problem is difficult for the following reasons:
        * not all the words in the display name have associated tags
        * the quantity field is stored as a number, but it appears
          as a string in the display name
        * the comment is often a compilation of different comments in
          the display name

    """
    ret = []

    # strip parens from the token, since they often appear in the
    # display_name, but are removed from the comment.
    token = normalizeToken(token)
    decimalToken = parse_numbers(token)

    # Iterate through the labels in descending order of label importance.
    for label_key in ['name', 'unit', 'qty', 'comment', 'range_end']:
        label_value = labels[label_key]
        if isinstance(label_value, str):
            initial_label_tokens = ingredientPhraseTokenizer.tokenize(
                label_value)

            final_label_tokens = []
            numberTokenGroup = []
            for label_token in initial_label_tokens:
                if bool(re.match('^[0-9/]+$', label_token)):
                    numberTokenGroup.append(label_token)
                else:
                    if len(numberTokenGroup):
                        if numberTokenGroup[0].isdigit():
                            original_text = matchNumberTokensToOriginalString(
                                numberTokenGroup, label_value)
                            if original_text == None:
                                final_label_tokens.append(
                                    "".join(numberTokenGroup))
                            else:
                                final_label_tokens.append(
                                    clumpFractions(original_text))
                        else:
                            for numberToken in numberTokenGroup:
                                final_label_tokens.append(numberToken)

                    numberTokenGroup = []

                    final_label_tokens.append(label_token)

            if len(numberTokenGroup):
                if numberTokenGroup[0].isdigit():
                    original_text = matchNumberTokensToOriginalString(
                        numberTokenGroup, label_value)
                    if original_text == None:
                        final_label_tokens.append("".join(numberTokenGroup))
                    else:
                        final_label_tokens.append(
                            clumpFractions(original_text))
                else:
                    for numberToken in numberTokenGroup:
                        final_label_tokens.append(numberToken)

            for n, vt in enumerate(final_label_tokens):
                if normalizeToken(vt) == token:
                    ret.append(label_key.upper())

        elif decimalToken is not None:
            if math.isclose(label_value, decimalToken):
                ret.append(label_key.upper())

    return ret


def add_prefixes(data):
    """
    We use BIO tagging/chunking to differentiate between tags
    at the start of a tag sequence and those in the middle. This
    is a common technique in entity recognition.

    Reference: http://www.kdd.cis.ksu.edu/Courses/Spring-2013/CIS798/Handouts/04-ramshaw95text.pdf
    """
    prevTags = None
    newData = []

    for n, (token, tags) in enumerate(data):

        newTags = []

        for t in tags:
            p = "B" if ((prevTags is None) or (t not in prevTags)) else "I"
            newTags.append("%s-%s" % (p, t))

        newData.append((token, newTags))
        prevTags = tags

    return newData


def best_tag(tags):

    if len(tags) == 1:
        return tags[0]

    # if there are multiple tags, pick the first which isn't COMMENT
    else:
        for t in tags:
            if (t != "B-COMMENT") and (t != "I-COMMENT"):
                return t

    # we have no idea what to guess
    return "OTHER"


def csv_file_2_examples(file_name):
    data = pd.read_csv(file_name)
    data = data.fillna("")

    examples = []
    for index, row in data.iterrows():
        examples.append(row_to_example(row))
        if index % 1000 == 0:
            print('Processed example: ' + str(index))

    return examples


# don't modify tags at all
class TagTokenizer(Tokenizer):
    def tokenize(self, s):
        s = tf.compat.as_text(s)
        return [s]


def build_encodings(examples):
    vocab_list = sorted(
        set([
            word for example in examples
            for word in ingredientPhraseTokenizer.tokenize(example[0])
        ]))
    tag_list = sorted(set([tag for example in examples for tag in example[1]]))

    word_encoder = TokenTextEncoder(vocab_list,
                                    tokenizer=ingredientPhraseTokenizer)
    tag_encoder = TokenTextEncoder(tag_list,
                                   oov_buckets=0,
                                   tokenizer=TagTokenizer())

    return word_encoder, tag_encoder


def build_dataset(examples, word_encoder, tag_encoder):
    def example_generator():
        for example in examples:
            # TODO: build a better encoder that doesn't require these
            # weird hacks - custom splitting, etc.
            yield (
                word_encoder.encode(example[0]),
                [tag_encoder.encode(tag)[0] for tag in example[1]],
            )

    return tf.data.Dataset.from_generator(example_generator,
                                          output_types=(tf.int32, tf.int32))


def load_examples(data_path, dataset_name):
    examples_path = data_path + f"/{dataset_name}_examples.json"

    # load examples if they've already been computed
    if (os.path.exists(examples_path)):
        with open(examples_path, "r") as f:
            examples = json.load(f)

    # otherwise compute and save examples from csv file
    else:
        examples = csv_file_2_examples(data_path + f"/{dataset_name}.csv")
        with open(data_path + f"/{dataset_name}_examples.json", "w") as f:
            json.dump(examples, f, indent=4)

    return examples


def preprocess(data_path):
    train_examples = load_examples(data_path, "train")
    dev_examples = load_examples(data_path, "dev")
    test_examples = load_examples(data_path, "test")

    word_encoder, tag_encoder = build_encodings(train_examples)
    '''
    encoding_path = data_path + "/encodings"
    word_encoder_path = encoding_path + "/word_encoder"
    tag_encoder_path = encoding_path + "/tag_encoder"
    if (os.path.exists(encoding_path)):
        word_encoder = TokenTextEncoder.load_from_file(word_encoder_path)
        tag_encoder = TokenTextEncoder.load_from_file(tag_encoder_path)
    else:
        os.mkdir(encoding_path)
        word_encoder, tag_encoder = build_encodings(train_examples)
        word_encoder.save_to_file(word_encoder_path)
        tag_encoder.save_to_file(tag_encoder_path)
    '''

    train_dataset = build_dataset(train_examples, word_encoder, tag_encoder)
    dev_dataset = build_dataset(dev_examples, word_encoder, tag_encoder)
    test_dataset = build_dataset(test_examples, word_encoder, tag_encoder)

    return train_dataset, dev_dataset, test_dataset, word_encoder, tag_encoder
