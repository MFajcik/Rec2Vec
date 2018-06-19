import argparse

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.nn.utils.rnn import pack_sequence

from dataprocessor import DataProcessor
from evaluation.analogy_questions import eval_analogy_questions

__author__ = "Martin Fajčík"


class Rec2Vec(nn.Module):
    def __init__(self, data_proc):
        super(Rec2Vec, self).__init__()
        self.data_processor = data_proc

        # TODO, move this away from init`z
        self.layers = 2
        self.hidden_size = data_proc.embedding_size

        self.in_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size,
                                          _weight=self.data_processor.original_embeddings)

        cell = nn.GRU if data_proc.use_gru else nn.LSTM
        self.rnn = cell(input_size=data_proc.embedding_size, num_layers=self.layers, hidden_size=self.hidden_size,
                        bidirectional=True)
        if self.data_processor.vembeddings is not None:
            print("Using preinitialized V-embeddings!")
            self.v_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size,_weight=self.data_processor.vembeddings)
        else:
            self.v_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size)
        self.logsigmoid = nn.LogSigmoid()

        self.initial_lr = args.learning_rate
        self.optimizer = optimizer.Adam(self.parameters(),
                                        lr=args.learning_rate)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
            self.in_embeddings.cuda()
            self.rnn.cuda()
            self.logsigmoid.cuda()

        self.hidden = self.init_hidden(directions=2)

    # Initializes hidden layer
    # initfun examples:
    # torch.randn initializes hidden cells with gaussian noise (0 mean, 1 variance)
    # torch.zeros initializes hidden cells with 0s
    # The axes semantics are (num_layers*directions, minibatch_size, hidden_dim)
    def init_hidden(self, directions, initfun=torch.randn):
        if self.data_processor.use_gru:
            hidden = initfun(self.layers * directions, self.data_processor.batch_size, self.hidden_size)
            return hidden.cuda() if self.use_cud else hidden
        else:
            hidden = initfun(self.layers * directions, self.data_processor.batch_size, self.hidden_size)
            cells = initfun(self.layers * directions, self.data_processor.batch_size, self.hidden_size)
            return (hidden.cuda(), cells.cuda()) if self.use_cuda else (hidden, cells)

    def forward(self, pos, seqlengths, targets, neg_v):
        # LSTM inpus has to have shape
        # (seq_len, batch, input_size

        all_in_embs = self.in_embeddings(pos)
        in_emb_seqs = pack_sequence(torch.split(all_in_embs, seqlengths, dim=0))

        # Output is in form of PackedSequence
        # Assume the following example
        # example is drawn in batch first dimension, but in reality
        # if we do not use batch first parameter, like here,
        # the seq_len is first
        # EXAMPLE:
        # Packed Sequence (188x2*dims)  ==> Padded sequence
        #    Tensor(18x2*dims)              Tensor(6xx2*dims)
        #    seq len
        #    4  4  3  3  2  2               4  4  3  3  2  2
        # b  ----------------          # b  ----------------
        # a  ----------------          # a  ----------------
        # t  ----------                # t  ----------000000
        # ch ----                      # ch ----000000000000

        output, lasthidden = self.rnn(in_emb_seqs, self.hidden)
        if not self.data_processor.use_gru:
            lasthidden = lasthidden[0]

        u_emb_batch = lasthidden[-1, :, :] + lasthidden[-2, :, :]
        # output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)  # unpack
        # Leave only last state# Sum bidirectional outputs
        v_emb_batch = self.v_embeddings(targets)

        # o is sigmoid function
        # NS loss for 1 sample and max objective is
        ##########################################################
        # log o(v^T*u) + sum of k samples log o(-negative_v^T *u)#
        ##########################################################
        # log o(v^T*u)  = score
        # sum of k samples log o(-negative_v^T *u) = neg_score

        # Multiply element wise
        score = torch.mul(u_emb_batch, v_emb_batch)
        # Sum so we get dot product for each row
        score = torch.sum(score, dim=1)
        score = self.logsigmoid(score)
        v_neg_emb_batch = self.v_embeddings(neg_v)
        # v_neg_emb_batch has shape [BATCH_SIZE,NUM_OF_NEG_SAMPLES,EMBEDDING_DIMENSIONALITY]
        # u_emb_batch has shape [BATCH_SIZE,EMBEDDING_DIMENSIONALITY]
        neg_score = torch.bmm(v_neg_emb_batch, u_emb_batch.unsqueeze(2))
        neg_score = self.logsigmoid(-1. * neg_score)

        if data_proc.visdom:
            self.loss_window = data_proc.visdom.line(X=torch.zeros((1,)).cpu(),
                                                     Y=torch.zeros((1)).cpu(),
                                                     opts=dict(xlabel='Bytes processed',
                                                               ylabel='Loss',
                                                               ytype="log",
                                                               title=f"R2Vec Training {type(self.optimizer).__name__}, lr={args.learning_rate}",
                                                               legend=['Loss']))

        return -1. * (torch.sum(score) + torch.sum(neg_score))

    def _train(self, previously_read=0, epoch=0):

        batch_gen = data_proc.create_batch_gen(previously_read)
        iteration = 0
        for sample in batch_gen:
            sample.sort(key=lambda t: len(t[0]), reverse=True)
            indices = [0]
            for p in sample:
                indices.append(len(p[0]))
            indices = indices[1:]

            # LSTM for embeddings before and after word
            targets = torch.LongTensor([item[1] for item in sample])
            pos = torch.LongTensor([item for sublist in sample for item in sublist[0]])
            neg_v = self.data_processor.get_neg_v_neg_sampling()
            neg_v = torch.LongTensor(neg_v)

            if self.use_cuda:
                pos = pos.cuda()
                targets = targets.cuda()
                neg_v = neg_v.cuda()
            self.optimizer.zero_grad()
            loss = self.forward(pos, indices, targets, neg_v)
            loss.backward()
            self.optimizer.step()

            if iteration % 200 == 0:
                if self.data_processor.visdom:
                    self.data_processor.visdom.line(
                        X=(torch.ones((1, 1)).cpu() * self.data_processor.bytes_read).squeeze(1),
                        Y=torch.Tensor([loss.data]).cpu(),
                        win=self.loss_window,
                        update='append')
                if iteration % 4000 == 0:
                    print(f"\nEpoch {epoch}, Loss: {loss.data}")
                    if self.data_processor.analogy_questions is not None:
                        print("#V_EMB")
                        eval_analogy_questions(data_processor=self.data_processor,
                                               embeddings=self.v_embeddings,
                                               use_cuda=self.use_cuda)
                        print("#ORIG_EMB")
                        eval_analogy_questions(data_processor=self.data_processor,
                                               embeddings=self.in_embeddings,
                                               use_cuda=self.use_cuda)

                    # if iteration % 50000 == 0:
                    #     print("\nSANITY CHECK")
                    #     print(
                    #         "----------------------------------------------------------------------------------------------------------------------------------")
                    #     for testword in self.data_processor.sanitycheck:
                    #         print(f"Nearest words to '{testword}' are: {', '.join(self.find_nearest(testword))}")
                    #     print(
                    #         "----------------------------------------------------------------------------------------------------------------------------------")

            iteration += 1
        return self.data_processor.bytes_read

    # The vec file is a text file that contains the word vectors, one per line for each word in the vocabulary.
    # The first line is a header containing the number of words and the dimensionality of the vectors.
    # Subsequent lines are the word vectors for all words in the vocabulary, sorted by decreasing frequency.
    # Example:
    # 218316 100
    # the -0.10363 -0.063669 0.032436 -0.040798...
    # of -0.0083724 0.0059414 -0.046618 -0.072735...
    # one 0.32731 0.044409 -0.46484 0.14716...
    def save(self, vec_path,embeddings):
        vocab_size = self.data_processor.vocab_size
        embedding_dimension = self.data_processor.embedding_size
        # Using linux file endings
        with open(vec_path, 'w') as f:
            print("Saving .vec file to {}".format(vec_path))
            f.write("{} {}\n".format(vocab_size, embedding_dimension))
            for word, id in self.data_processor.w2id.items():
                tensor_id = torch.LongTensor([id])
                if self.use_cuda:
                    tensor_id = tensor_id.cuda()
                embedding = embeddings(tensor_id).cpu().squeeze(0).detach().numpy()
                f.write("{} {}\n".format(word, ' '.join(map(str, embedding))))


def init_parser(parser):
    parser.add_argument("-c", "--corpus", help="input data corpus", required=True)
    parser.add_argument("--vocab", help="precalculated vocabulary")
    parser.add_argument("--eval_aq", "--eval_analogy_questions",
                        help="file with analogy questions to do the evaluation on", default=None)
    parser.add_argument("-v", "--verbose", help="increase the model verbosity", action="store_true")
    parser.add_argument("-w", "--window", help="size of a context window",
                        default=5)
    parser.add_argument("-ns", "--nsamples", help="number of negative samples",
                        default=25)
    parser.add_argument("-mf", "--min_freq", help="minimum frequence of occurence for a word",
                        default=5)
    parser.add_argument("-lr", "--learning_rate", help="initial learning rate",
                        default=0.025
                        )
    parser.add_argument("-d", "--dimension", help="size of the embedding dimension",
                        default=300)
    parser.add_argument("-br", "--bytes_to_read", help="how much bytes to read from corpus file per chunk",
                        default=512)
    parser.add_argument("-bs", "--batch_size", help="size of 1 batch in training iteration", default=512)
    parser.add_argument("-pc", "--phrase_clustering",
                        help="enable phrase clustering as described by Mikolov (i.e. New York becomes New_York)",
                        default=True)
    parser.add_argument("-e", "--embeddings",
                        help="embeddings to be encoded, without any positional or ordering information.")
    parser.add_argument("-ve", "--vembeddings",
                        help="pretrained V matrix embeddings, without any positional or ordering information, used for preinitialisation.",
                        default=None)
    parser.add_argument("-ri", "--random_ints",
                        help="how many random ints for window subsampling to precalculate at once",
                        default=1310720  # 5 megabytes of int32s
                        )
    parser.add_argument("-tr", "--subsfqwords_tr", help="subsample frequent words threshold", default=1e-4)
    parser.add_argument("--visdom", help="visualize training via visdom_enabled library", default=True)
    parser.add_argument("--gru", help="use GRU units instead of LSTM units", default=False)
    parser.add_argument("--sanitycheck",
                        help='list of words for which the nearest word embeddings are found during training, '
                             'serves as sanity check, i.e. "dog family king eye"',
                        default="dog family king eye")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    data_proc = DataProcessor(args)
    model = Rec2Vec(data_proc)

    bytes_read = 0
    epochs = 30
    for e in range(epochs):
        print(f"Starting epoch: {e}")
        bytes_read = model._train(previously_read=bytes_read, epoch=e)

    model.save(f"trained/Trefined_Oembeddings{epochs}.vec",model.in_embeddings)
    model.save(f"trained/Trefined_Vembeddings{epochs}.vec",model.v_embeddings)