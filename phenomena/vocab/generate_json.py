import json
import csv
import os
import random
from vocab_utils import (
    get_nouns, get_nouns2, get_verbs, get_verbs2,
    get_adv1s, get_adv2s, get_preposition_nouns, get_prepositions,
    get_verbs_irregular
)

NOUNS = get_nouns()
NOUNS2 = get_nouns2()
VERBS = get_verbs_irregular()
VERBS2 = get_verbs_irregular()
ADVS = get_adv1s()
ADVS2 = get_adv2s()
PREPOSITIONS = get_prepositions()
PREPNOUNS = get_preposition_nouns()

def make_patch_prefix(clean_prefix, is_within_rc=False):
    words = clean_prefix.split()
    if not is_within_rc:
        noun = words[1]
        for noun_pair in NOUNS:
            noun_s, noun_p = noun_pair
            if noun == noun_s:
                words[1] = noun_p
                break
            elif noun == noun_p:
                words[1] = noun_s
                break
    else:
        noun2 = words[-1]
        for noun_pair in NOUNS2:
            noun_s, noun_p = noun_pair
            if noun2 == noun_s:
                words[-1] = noun_p
                break
            elif noun2 == noun_p:
                words[-1] = noun_s
                break
    return " ".join(words)


def replace_with_irregular(verb):
    random.seed(12)
    irregular_pair = random.choice(VERBS2)
    # detect number of verb
    all_verb_pairs = VERBS
    all_verb_pairs.extend(VERBS2)
    for verb_pair in all_verb_pairs:
        verb_s, verb_p = verb_pair
        if verb == verb_s:
            return irregular_pair[0]
        elif verb == verb_p:
            return irregular_pair[1]
        
    raise Exception("Could not find verb in verb lists.")


for filename in os.listdir("."):
    if not filename.endswith(".txt"):
        continue
    out_filename = filename.split(".txt")[0] + ".json"
    if "within_rc" in filename:
        is_within_rc = True
    else:
        is_within_rc = False

    with open(filename, 'r') as in_data, open(out_filename, 'w') as out_data:
        # Set up variables
        correct_sentence = None
        incorrect_sentence = None
        # Iterate through examples
        reader = csv.reader(in_data, delimiter="\t")
        for row in reader:
            sentence, case, grammaticality, _id = row
            if grammaticality == "correct":
                correct_sentence = sentence
            else:
                incorrect_sentence = sentence
            if correct_sentence is None or incorrect_sentence is None:
                continue

            clean_prefix = " ".join(correct_sentence.split()[:-1])
            patch_prefix = make_patch_prefix(clean_prefix,
                                             is_within_rc=is_within_rc)
            
            clean_answer = replace_with_irregular(correct_sentence.split()[-1])
            patch_answer = incorrect_sentence.split()[-1]

            data = {"clean_prefix": clean_prefix, "patch_prefix": patch_prefix,
                    "clean_answer": clean_answer, "patch_answer": patch_answer,
                    "case": case}
            out_data.write(json.dumps(data) + "\n")
            correct_sentence = None
            incorrect_sentence = None