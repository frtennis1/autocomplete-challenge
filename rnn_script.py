import re
import sys
from sys import stdin, stderr, stdout

import pickle
import string
import torch

from generate import *

# see training file for how decoder was trained
stderr.write('Loading pickle')
decoder = torch.load('rnn_model.pt')

sep = chr(31)

def predict_review():
    review_so_far = ''
    hidden = None
    eliminated = []
    preds = []

    while True:
        next_chars = stdin.readline().rstrip('\r\n')
        if len(next_chars) == 0: 
            return # EOM

        eliminated = [s[len(next_chars):] for s in eliminated + preds
            if s[:len(next_chars)] == next_chars]
            

        review_so_far += next_chars

        preds, hidden = predict(review_so_far, decoder, hidden=hidden,
            hidden_lag=len(next_chars), eliminated=eliminated,
            output_hidden=True)

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

