Coatue Movie Review Challenge
=============================

Submission by Francisco Rivera.

Installing
----------

The project is written in Python 3.6, and requires the installations of a couple
of libraries. One way to do this is to have `pip` and run

```
pip install -r requirements.txt
```
Note that this will install a version of PyTorch compatible with Linux. If,
instead, the program will be run on a Mac, one can instead install,

```
pip install -r requirements-mac.txt
```
For any other system, see <pytorch.org> for information on the installation
file.

Training
--------

The model requires relatively lengthy training in order to properly learn the
what characters are likely to come next. Training can be accomplished by running

```
python train.py <fname>
```

where `<fname>` is the filename of the text reviews on which to train. Note that
the model does not consider the meta-data in making predictions. Furthermore,
this script can be run with options passed as command-line arguments in the
interest of generality, but the defaults are the ones the submitted model was
trained with.


Running
-------

In order to grade the submission, simply run

```
python grader.py python rnn_script.py
```

Note that the location of the model needs to be hard-coded in `rnn_script.py`.
Currently, it is set to be `rnn_model.pt`, which is the pre-trained model
included with the submission.

In order to run the benchmark, you can run,

```
python grader.py python trie_script.py
```

Approach
--------

Recurrent neural networks (RNN) have proven extraordinarily effective at a host
of tasks, among them text-prediction. In this project, we use one to aid in the
autocompletion of movie reviews. This set-up has two main components: the RNN
engine that predicts future characters given past ones, and a decision engine to
output three predictions. First, we discuss each component separately, and then
we conclude with a discussion on how this approach fares to another less
sophisticated benchmark.

The RNN engine concerns itself only with predicting the distribution of the next
characters given all the characters that precede it. It is trained on a cleaned
random sample of the reviews. The cleaned dataset restricts the possible
characters to a relatively large set of possibilities, and transforms all the
letters to lowercase. It is trained using cross-entropy loss, and as such gives
a probability distribution over the next character. Two architectures were
considered for the RNN: an LSTM (Long Short Term Memory) and GRU (Gated
Recurrent Units). LSTM's have had the most spectacular results in the
literature, being the most widely used for NLP purposes. Unfortunately, given
the computational and time constraints, and LSTM did not achieve as good as
success as the GRU, a simpler cousin which shares many commonalities. Training
the final model used takes under an hour on a Macbook Pro. Further experimenting
is possible with the particular configuration of the GRU, but because of
training time-constraints, a vanilla configuration that worked reasonably was
used.

The decision engine queries the RNN engine for the probability distributions of
characters to come. It takes these probabilities as truth, and attempts to
select three character sequences which maximize the expectation of the number of
characters in the selected sequence (taken to be 1 if none of them are correct).
This aligns with the metric whereby the prediction engine will be judged.
However, querying the RNN is a mildly costly operation. Because of this, a
simplication is considered: we instead select the three character sequences
which individually maximize the expectation of the number of characters in the
correct sequence. This is different from the ground-truth metric we care about
because whether a sequences are correct aren't independent events. For instance,
if I observe a "t" and predict "he" and "hem" as my sequences, if I know that
"he" is not correct, then "hem" will surely not be correct either. If it happens
that the top subsequences are highly associated (which empirically, they appear
to be), this means that the algorithm does not choose different enough strings.
Nevertheless, for computational tractability, we run this decision rule which we
find performs well in practice.

In addition, the decision engine remembers past guesses that have not been
correct which discount future guesses, and knows to avoid those. For instance,
if I see the letter "a," guess "mazing," observe that this is wrong, and now
have to make a prediction for "am," the engine knows to not guess "azing" or
anything that starts with this.

When considering how this does, we need a benchmark. To this end, we follow the
suggestion of the problem spec and implement a trie which stores the frequency
of words in the training set, and predicts the highest-likelihood word given the
characters since the last space or punctuation. Three-and-smaller-prefixes are
cached in a separate dictionary to avoid the most expensive calls to the trie.
The example code takes approximately 78,000 function calls in the default
validation-set, whereas the word-prefix approach takes only 58,000. Furthermore,
the RNN takes just over 51,000, adding perceptible value beyond the memorization
of common words.

Future Work
-----------

However, we argue that the biggest value of the RNN strategy is not the
moderate performance boost already demonstrated, but rather the potential for
future extendability. We elaborate both on this perceived strength as well as
potential avenues for future development here.

Frankly, the length of training that went into the submitted model is laughable.
It is more than evident that convergence was not reached in that time. Not only
that, computational resources precluded more complex models, which machine
learning literature have shown can be tremendously successful.  Even training
this for a couple of days or using a GPU could lead to tremendous gains without
any engineer-hours.

In addition, the framework of an RNN is tremendously flexible. For instance, the
meta-data of a particular review could be fed into the RNN as concatenated
tensors allowing the RNN to incorporate this information. If a new data-set
comes along on a different subject matter or a different language, the RNN
framework would be directly transferable, simply with new training. This
contrasts with other approaches that could be more focused on eeking out
additional performance from expensively-computed (in terms of engineer-hours)
features that may translate poorly to other domains.

In addition to a more complex neural network architecture or the addition of
more inputs such as meta-data, the RNN component may benefit from a dual
architecture. In essense, there can be two RNNs: one that attempts to predict
the next word using something such as the GloVe embeddings, and one that
attempts to predict the next character, such as what is currently implemented.
This dual nature could further bolster the ability of the neural network to
predict longer string, increasing its performance.

Finally, with regards to the decision engine, refining the approximation used
may significantly improve performance. While it is likely intractable to get the
precise optimum, implementing a heuristic to avoid similar strings (particularly
those where one is a prefix of the other), could itself be a step in the right
direction. 

Appendix: File Explanations
---------------------------
- `generate.py` houses the functions that compute the distribution of the next
  character as well as the decision logic to select the three most promising
  sequences.

- `grader.py` is the untouched provided grading file.

- `helpers.py` provides some straightforward helper functions.

- `model.py` houses the definition and architecture of the RNN.

- `trie_script.py` is the benchmark word-completion implementation.

- `rnn_model.pt` saves the architecture and weights of the trained model.

- `short_d.pickle` is the cache dictionary of word prefixes.

- `word_trie.pickle` saves the trie datastructure 

- The Jupyter notebooks contain scratch work that was used during the creation
  of the system. They are mostly included for the sake of completeness, and no
  assurances are made that all the cells will properly run.

