#!/usr/bin/env python
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer


part1 = r'@[A-Za-z0-9]+'
part2 = r'http?//[^ ]+'
combined_pat = r'|'.join((part1, part2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't": "is not",
                  "aren't": "are not",
                  "wasn't": "was not",
                  "weren't": "were not",
                  "haven't": "have not",
                  "hasn't": "has not",
                  "hadn't": "had not",
                  "won't": "will not",
                  "wouldn't": "would not",
                  "don't": "do not",
                  "doesn't": "does not",
                  "didn't": "did not",
                  "can't": "can not",
                  "couldn't": "could not",
                  "shouldn't": "should not",
                  "mightn't": "might not",
                  "mustn't": "must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def word_cleaner_updated(text):
    soup = BeautifulSoup(text, 'html_parser')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()
