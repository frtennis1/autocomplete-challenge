import re
import sys
from sys import stdin, stderr, stdout

import pygtrie as trie
import pickle
import string

# see Jupyter notebook for how the Trie was created
stderr.write('Loading pickle')
short_d = pickle.load(open('short_d.pickle', 'rb'))
word_trie = pickle.load(open('word_trie.pickle', 'rb'))
stderr.write('Done with pickle')
interesting = string.ascii_lowercase + "'"

sep = chr(31)

def predict(text):
    # we need to predict here
    last_word = text.split(' ')[-1]
    clean_word = ''.join([c for c in last_word if c in interesting])

    if clean_word == '':
        top_words = ["there's", 'there', 'the']

    elif len(clean_word) < 4 and clean_word in short_d.keys():
        top_words = [w[len(clean_word):] for w in short_d[clean_word]]

    elif word_trie.has_subtrie(clean_word):
        top_words = [x[0][len(clean_word):] for x in sorted(
            word_trie.items(clean_word)[:50],
            key=lambda x:x[1],
            reverse=True)[:3]]
    else:
        top_words = [' ', '  ', '   ']

    if len(top_words) < 3:
        top_words = top_words + [' ',]*(3-len(top_words))
    
    # return top_words
    return [w + ' ' for w in top_words]


def predict_review():
    review_so_far = ''
    while True:
        next_chars = stdin.readline().rstrip('\r\n')
        if len(next_chars) == 0: 
            return # review EOM

        review_so_far += next_chars

        preds = predict(review_so_far)

        try:
            print('{}{}{}{}{}'.format(preds[0], sep, preds[1], sep, preds[2]))
            stdout.flush()
        except Exception:
            print(' {} {} '.format(sep, sep))
            stdout.flush()


while True:
    try:
        metadata = stdin.readline().rstrip('\r\n').split(',')
        product_id, user_id, unix_date, rating = metadata
        predict_review()
    except EOFError:
        sys.exit(0)
    except Exception as e:
        print(e, file=stderr)
        sys.exit(1)
