import torch
import os
import argparse
import heapq

from helpers import *
from model import *

# from models import decoder, lstm, rnn

def get_hidden_state(decoder, s, hidden=None):
    if hidden is None:
        hidden = decoder.init_hidden(1)
    
    s_input = Variable(char_tensor(s).unsqueeze(0))
    
    for p in range(len(s) - 1):
        _, hidden = decoder(s_input[:,p], hidden)
    
    return hidden
    

def distribution(decoder, prime_str='A', temperature=.8, hidden=None,
        output_hidden=False, bad=[]):

    if hidden is None:
        hidden = decoder.init_hidden(1)
    
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    # Use priming string to "build up" hidden state

    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    output, hidden = decoder(inp, hidden)
    
    # Sample from the network as a multinomial distribution
    distr = output.data.view(-1).div(temperature).exp()
    if bad:
        distr[torch.LongTensor(bad)] = 0

    if not output_hidden:
        return distr / distr.sum()
    else:
        return distr / distr.sum(), hidden 

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))

    return predicted


def predict(s,
         decoder,
         hidden=None,
         hidden_lag=1,
         eliminated=[],
         output_hidden=False):
    """
    If hidden state is given, it should not be for the full string
    """
    probs = [{'': 1},]
    scores = [(0,''),(0, ' '), (0,'e')]
    heapq.heapify(scores)

    is_useful = True
    while len(probs) <= 2 or is_useful:
        new_guesses = []
        for guess in probs[-1].keys():
            bad_chars = set([elim[len(guess)] for elim in eliminated if
                             elim.startswith(guess) and
                             len(elim) > len(guess)])
            bad_i= [string.printable.find(x) for x in bad_chars]

            running_str = s + guess
            if guess != '':
                srtd, indices = torch.topk(distribution(
                    decoder, prime_str=guess, hidden=hidden, bad=bad_i), 3)
            else:
                if hidden is not None:
                    distr, hidden = distribution(decoder,
                                                 prime_str=s[-hidden_lag:], 
                                                 hidden=hidden,
                                                 output_hidden=True,
                                                 bad=bad_i)
                else:
                    distr, hidden = distribution(decoder,
                                                 s,
                                                 output_hidden=True,
                                                 bad=bad_i)
                srtd, indices = torch.topk(distr, 3)

            for val, i in zip(srtd, indices):
                if len(new_guesses) == 3:
                    heapq.heappushpop(new_guesses,
                        (probs[-1][guess] * val, guess + string.printable[i]))
                else:
                    heapq.heappush(new_guesses,
                        (probs[-1][guess] * val, guess + string.printable[i]))
        probs.append({})
        is_useful = False
        for val, guess in new_guesses:
            probs[-1][guess] = val
            if val*(len(guess)-1) > scores[0][0]:
                heapq.heappushpop(scores, (val*(len(guess)-1), guess))
                is_useful = True
    
    if not output_hidden:
        return [x[1] for x in scores]
    else:
        return [x[1] for x in scores], hidden
