#!/usr/bin/env python
# coding: utf-8

import pandas as pd

PATH = './'

simple = pd.read_csv(PATH + 'simple.txt', sep=' |\t', 
                     engine='python', 
                     names=['The','noun','verb','number',
                            'grammaticality','id'])

nounpp = pd.read_csv(PATH + 'nounpp.txt', sep=' |\t|_', 
                     engine='python', 
                     names=['The', 'noun', 'preposition', 'the',
                            'pp_noun', 'verb', 'n_number',
                            'pp_number', 'grammaticality', 'id'])

adv = pd.read_csv(PATH + 'adv_conjunction.txt', sep=' |\t|_', engine='python',
        names=['The','noun','adv1', 'and', 'adv2','verb','number',
            'grammaticality','id'])

rc = pd.read_csv(PATH + 'rc.txt', sep=' |\t|_', engine='python',
        names=['The','noun','that','the','noun2','verb2','verb','n1_number',
            'n2_number', 'grammaticality','id'])

within_rc = pd.read_csv(PATH + 'within_rc.txt', sep=' |\t|_', engine='python',
        names=['The','noun','that','the','noun2','verb','n1_number','n2_number',
            'grammaticality','id'])


# Construct nouns
n_singular = nounpp['noun'][nounpp['n_number'] == 'singular']\
        .drop_duplicates().reset_index(drop=True)
n_plural = nounpp['noun'][nounpp['n_number'] == 'plural']\
        .drop_duplicates().reset_index(drop=True)
n_frame = {'n_singular':n_singular, 'n_plural':n_plural}
nouns = pd.DataFrame(n_frame)

n2_singular = within_rc['noun2'][within_rc['n2_number'] == 'singular']\
        .drop_duplicates().reset_index(drop=True)
n2_plural = within_rc['noun2'][within_rc['n2_number'] == 'plural']\
        .drop_duplicates().reset_index(drop=True)
n2_frame = {'n2_singular': n2_singular, 'n2_plural': n2_plural}
nouns2 = pd.DataFrame(n2_frame)


# Construct verbs
v_singular = nounpp['verb'][nounpp['n_number'] == 'singular']\
        [nounpp['grammaticality'] == 'correct'].drop_duplicates()\
        .reset_index(drop=True)
v_plural = nounpp['verb'][nounpp['n_number'] == 'singular']\
        [nounpp['grammaticality'] == 'wrong'].drop_duplicates()\
        .reset_index(drop=True)
v_frame = {'v_singular':v_singular, 'v_plural':v_plural}
verbs = pd.DataFrame(v_frame)

v2_singular = rc['verb2'][rc['n2_number'] == 'singular']\
        .drop_duplicates().reset_index(drop=True)
v2_plural = rc['verb2'][rc['n2_number'] == 'plural']\
        .drop_duplicates().reset_index(drop=True)
v2_frame = {'v2_singular': v2_singular, 'v2_plural': v2_plural}
verbs2 = pd.DataFrame(v2_frame)

# Construct prepositional nouns
ppn_singular = nounpp['pp_noun'][nounpp['pp_number'] == 'singular']\
        .drop_duplicates().sort_values().reset_index(drop=True)

ppn_plural = nounpp['pp_noun'][nounpp['pp_number'] == 'plural']\
        .drop_duplicates().sort_values().reset_index(drop=True)

ppn_frame = {'ppn_singular':ppn_singular, 'ppn_plural':ppn_plural}

ppns = pd.DataFrame(ppn_frame)


# Construct prepositions
prepositions = nounpp['preposition'].drop_duplicates()

# Construct adverbs
adv1s = adv['adv1'].drop_duplicates()
adv2s = adv['adv2'].drop_duplicates()


def get_nouns():
    return [(s,p) for s, p in zip(nouns['n_singular'],
                                  nouns['n_plural'])]

def get_nouns2():
    return [(s,p) for s, p in zip(nouns2['n2_singular'],
                                  nouns2['n2_plural'])]

def get_verbs():
    return [(s,p) for s, p in zip(verbs['v_singular'], 
                                  verbs['v_plural'])]

def get_verbs2():
    return [(s,p) for s, p in zip(verbs2['v2_singular'],
                                  verbs2['v2_plural'])]

def get_preposition_nouns():
    return [(s,p) for s, p in zip(ppns['ppn_singular'], 
                                  ppns['ppn_plural'])]

def get_verbs_irregular():
    verbs = []
    with open("wordlists/verb_irregular.txt", 'r') as verblist:
        for verb in verblist:
            s, p = verb.strip().split()
            verbs.append((s, p))
    return verbs

def get_prepositions():
    return prepositions.tolist()


def make_template(noun, preposition, ppn):
    return ' '.join([noun, preposition, 'the', ppn])

def get_adv1s():
    return adv1s.tolist()

def get_adv2s():
    return adv2s.tolist()

if __name__ ==  '__main__':
    print(get_adv1s())
    print(get_adv2s())
