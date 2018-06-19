import math
import os
import pickle

from evaluation.analogy_questions import read_analogies
from nlpfit.preprocessing.nlp_io import read_word_lists

import visdom
import numpy as np
import time
from collections import deque


class DataProcessor():
    def __init__(self, args):
        self.min_freq = args.min_freq
        self.bytes_to_read = args.bytes_to_read
        self.corpus = args.corpus
        self.vocab_path = args.vocab
        self.batch_size = args.batch_size
        self.window_size = args.window
        self.thresold = args.subsfqwords_tr
        self.randints_to_precalculate = args.random_ints
        self.nsamples = args.nsamples
        self.embedding_size = args.dimension
        self.sanitycheck = args.sanitycheck.split()
        self.visdom_enabled = args.visdom

        self.use_gru = args.gru

        with open(args.embeddings, mode="rb") as f:
            print("Loading word embeddings...")
            self.original_embeddings = pickle.load(f)

        if args.vembeddings:
            print("Loading pretrained V word embeddings...")
            with open(args.vembeddings, mode="rb") as f:
                self.vembeddings = pickle.load(f)
        else:
            self.vembeddings = None

        if self.visdom_enabled:
            self.visdom = visdom.Visdom()

        # Load corpus vocab, and calculate prerequisities
        self.frequency_vocab_with_OOV = self.load_vocab() if args.vocab else self.parse_vocab()
        self.corpus_size = self.calc_corpus_size()
        # Precalculate term used in subsampling of frequent words
        self.t_cs = self.thresold * self.corpus_size

        self.frequency_vocab = self.calc_frequency_vocab()
        self.vocab_size = len(self.frequency_vocab) + 1  # +1 For unknown

        self.sample_table = self.init_sample_table()

        # Create id mapping used for fast U embedding matrix indexing
        self.w2id = self.create_w2id()
        self.id2w = {v: k for k, v in self.w2id.items()}

        # Preload eval analogy questions
        if args.eval_aq:
            self.eval_data_aq = args.eval_aq
            self.analogy_questions = read_analogies(file=self.eval_data_aq, w2id=self.w2id)

        self.cnt = 0
        self.benchmarktime = time.time()
        self.bytes_read = 0

    # For fast negative sampling
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8

        # Create proper uniform distribution raised on 3/4
        pow_frequency = np.array(list(self.frequency_vocab.values())) ** 0.75
        normalizer = sum(pow_frequency)
        normalized_freqs = pow_frequency / normalizer

        # Calculate how much table cells should each distribution element have
        table_distribution = np.round(normalized_freqs * sample_table_size)

        # Create vector table, holding number of items with element ID proprotional
        # to element id's probability in distribution\

        for wid, c in enumerate(table_distribution):
            self.sample_table += [wid] * int(c)
        return np.array(self.sample_table)

    def get_neg_v_neg_sampling(self):
        neg_v = np.random.choice(
            self.sample_table, size=(self.batch_size, self.nsamples)).tolist()
        return neg_v

    # This formula is not exactly the one from the original paper,
    # but it is inspired from tensorflow/models skipgram implementation.
    # The shape of this subsampling function is in fact similar, but
    # it's new behavior now adds relation to the corpus size to the formula
    # and also "it works with the large numbers" from frequency vocab
    # Also see my SO question&answer: https://stackoverflow.com/questions/49012064/skip-gram-implementation-in-tensorflow-models-subsampling-of-frequent-words
    def should_be_subsampled(self, w):
        f = self.frequency_vocab_with_OOV[w]
        keep_prob = (np.sqrt(f / self.t_cs) + 1.) * (self.t_cs / f)
        roll = np.random.uniform()
        return not keep_prob > roll

    def create_batch_gen(self, previously_read=0):
        fsize = os.path.getsize(self.corpus)
        # Create word list generator
        wordgen = read_word_lists(self.corpus, bytes_to_read=self.bytes_to_read, report_bytesread=True)
        # Create queue of random choices
        rchoices = deque(np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
        # create doubles
        word_from_last_list = []
        window_datasamples = []
        si = 0
        for wlist_ in wordgen:
            wlist = wlist_[0]
            self.bytes_read = wlist_[1]  # + previously_read

            # print(word_pairs)
            self.cnt += 1
            if self.cnt % 5000 == 0:
                t = time.time()
                p = t - self.benchmarktime
                # Derive epoch from bytes read
                total_size = fsize * (math.floor(self.bytes_read / fsize) + 1)
                print(
                    f"Time: {p/60:.2f} min - epoch state {self.bytes_read/total_size *100:.2f}% ({int(self.bytes_read/p/1e3)} KB/s)")

            # Discard words with min_freq or less occurences
            # Subsample of Frequent Words
            # hese words are removed from the text before generating the contexts
            wlist_clean = []
            for w in wlist:
                try:
                    if not (self.frequency_vocab_with_OOV[w] < self.min_freq or self.should_be_subsampled(w)):
                        wlist_clean.append(w)
                except KeyError as e:
                    print("Encountered unknown word!")
                    print(e)
                    print(f"Wlist: {wlist}")
            wlist = wlist_clean

            if not wlist:
                return

            wlist = list(map(lambda x: self.w2id[x], wlist))
            wlist = word_from_last_list + wlist
            word_from_last_list = []
            for i in range(si, len(wlist)):
                # if the window exceeds the buffered part
                if (i + self.window_size > len(wlist) - 1):
                    # find index m, that points on leftmost word still in a window
                    # of central word
                    m = max(i - self.window_size, 0)

                    # save the index of central word, with respect to start at leftmost word at position m
                    si = i - m

                    # throw away words before leftmost word, they have already been processed
                    word_from_last_list = wlist[m:]
                    break
                if not rchoices:
                    rchoices = deque(
                        np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
                r = rchoices.pop()
                if i - r < 0:
                    continue
                window_datasamples.append((wlist[i - r:i] + wlist[i + 1:i + r + 1], wlist[i]))

            if len(window_datasamples) > self.batch_size:
                yield window_datasamples[:self.batch_size]
                window_datasamples = window_datasamples[self.batch_size:]

    def load_vocab(self):
        from nlpfit.preprocessing.tools import read_frequency_vocab
        print("Loading vocabulary...")
        return read_frequency_vocab(self.vocab_path)

    def parse_vocab(self):
        # TODO: implement
        pass

    def calc_corpus_size(self):
        return sum(self.frequency_vocab_with_OOV.values())

    def calc_frequency_vocab(self):
        fvocab = dict()
        fvocab['UNK'] = 0
        for k, v in self.frequency_vocab_with_OOV.items():
            if v >= self.min_freq:
                fvocab[k] = v
        return fvocab

    def create_w2id(self):
        w2id = dict()
        for i, k in enumerate(self.frequency_vocab, start=0):
            w2id[k] = i
        return w2id
