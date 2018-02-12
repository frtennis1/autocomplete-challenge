import pandas as pd
import subprocess
import sys

from tqdm import tqdm


def load_reviews(i):
    return [review.strip() for review in open('reviews_{}.txt'.format(i), 'r')]


def load_metadata(i):
    colnames = ['product_id', 'user_id', 'unix_date', 'rating']
    return pd.read_csv('metadata_{}.csv'.format(i), header=None, names=colnames)


holdout_data = load_reviews(9)
holdout_metadata = load_metadata(9)
sep = chr(31)

invocations = 1
# Usage: python grader.py [command to invke your runner here]
#   In Python: python grader.py python -u runner.py
#   In Java: javac CoatueCaseStudy.java && python grader.py java CoatueCaseStudy
#
process = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for i in tqdm(range(100)): # use the first 100 reviews in the 10th file to sanity check against.
    review = holdout_data[i].lower()
    metadata = holdout_metadata.ix[i]
    metadata_str = '{},{},{},{}\n'.format(metadata.product_id, metadata.user_id, metadata.unix_date, metadata.rating)
    process.stdin.write(bytes(metadata_str, 'utf8'))
    process.stdin.flush()

    j = 1
    # Type the first character and start hunting for predictions
    process.stdin.write(bytes('{}\n'.format(review[0]), 'utf8'))
    process.stdin.flush()
    stdoutdata = process.stdout.readline().decode('utf8').rstrip('\r\n')
    while True:
        next_chars = 1
        try:
            review_typed, review_to_type = (review[:j], review[j:])
            pred1, pred2, pred3 = sorted(stdoutdata.lower().split(sep), key=len, reverse=True)
            if review_to_type.startswith(pred1) and pred1 != '':
                next_chars = len(pred1)
            elif review_to_type.startswith(pred2) and pred2 != '':
                next_chars = len(pred2)
            elif review_to_type.startswith(pred3) and pred3 != '':
                next_chars = len(pred3)
            else:
                next_chars = 1
        except IndexError as e:
            next_chars = 1

        if j + next_chars >= len(review):
            process.stdin.write(bytes('\n', 'utf8'))
            process.stdin.flush()
            break # the review has been completely typed
        process.stdin.write(bytes('{}\n'.format(review[j:j + next_chars]), 'utf8'))
        process.stdin.flush()
        stdoutdata = process.stdout.readline().decode('utf8').rstrip('\r\n')
        j += next_chars
        invocations += 1


process.terminate()
print('Terminated in {} invocations of predict.'.format(invocations))

