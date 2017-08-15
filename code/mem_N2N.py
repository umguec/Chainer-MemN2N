from chainer import Chain, optimizers, serializers
import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import collections
import math
import numpy as np
import pickle
import random
import time
import tqdm

class MemN2N(object):
    def __init__(self, corpus, words, **kwargs):
        self.batchsize = kwargs.get('batchsize', 128)
        self.epochs = kwargs.get('epochs', 100)
        self.kwargs = kwargs
        self.log = {
            ('training', 'loss') : [],
            ('training', 'perplexity') : [],
            ('training', 'throughput') : [],
            ('validation', 'loss') : [],
            ('validation', 'perplexity') : [],
            ('validation', 'throughput') : []
        }
        self.memsize = kwargs.get('memsize', 100)
        self.model = Model(
            kwargs.get('edim', 150),
            kwargs.get('init_hid', 0.1),
            kwargs.get('init_std', 0.05),
            kwargs.get('lindim', 75),
            kwargs.get('memsize', 100),
            kwargs.get('nhop', 6),
            words
        )
        self.optimizer = optimizers.SGD(kwargs.get('sdt', 0.01))
        self.vocabulary = Vocabulary(corpus, words)
        self.xp = np

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(
            chainer.optimizer.GradientClipping(kwargs.get('maxgradnorm', 50))
        )

    def _run(self, X):
        time_ = time.time()
        idx = range(self.memsize, len(X))
        loss = 0

        if chainer.config.train:
            random.shuffle(idx)

        for i in tqdm.tnrange(0, len(idx), self.batchsize, leave = False):
            x = []
            a = []

            for j in xrange(i, min(i + self.batchsize, len(idx) - 1)):
                x.append(X[idx[j] - self.memsize : idx[j]])
                a.append(X[idx[j]])

            x = self.xp.array(x, 'int32')
            a_hat = self.model(x)
            a = self.xp.array(a, 'int32')
            loss_ = a_hat.shape[0] * F.softmax_cross_entropy(a_hat, a)

            if chainer.config.train:
                self.model.cleargrads()
                loss_.backward()
                self.optimizer.update()

            loss += float(loss_.data)

        loss /= len(idx)
        perplexity = math.exp(loss)
        throughput = len(idx) / (time.time() - time_)

        return loss, perplexity, throughput

    @classmethod
    def load(cls, prefix):
        with open('{}dump.p'.format(prefix), 'rb') as f:
            dump = pickle.load(f)

        memN2N = cls(None, len(dump['id']), **dump['kwargs'])
        memN2N.log = dump['log']

        serializers.load_npz('{}model.npz'.format(prefix), memN2N.model)
        serializers.load_npz('{}optimizer.npz'.format(prefix), memN2N.optimizer)

        memN2N.optimizer.lr = dump['lr']
        memN2N.vocabulary.id = dump['id']
        memN2N.vocabulary.word = dump['word']

        return memN2N

    def save(self, prefix):
        dump = {
            'id': self.vocabulary.id,
            'kwargs': self.kwargs,
            'log': self.log,
            'lr': self.optimizer.lr,
            'word': self.vocabulary.word
        }

        with open('{}dump.p'.format(prefix), 'wb') as f:
            pickle.dump(dump, f, -1)

        serializers.save_npz('{}model.npz'.format(prefix), self.model)
        serializers.save_npz('{}optimizer.npz'.format(prefix), self.optimizer)

    def test(self, X):
        with chainer.using_config('train', False):
            loss, perplexity, throughput = self._run(X['test'])

        return loss, perplexity, throughput

    def train_and_validate(self, X, callback = None):
        loss_ = float('nan')

        for i in tqdm.tnrange(self.epochs):
            for j in ['training', 'validation']:
                with chainer.using_config('train', j == 'training'):
                    loss, perplexity, throughput = self._run(X[j])

                self.log[j, 'loss'].append(loss)
                self.log[j, 'perplexity'].append(perplexity)
                self.log[j, 'throughput'].append(throughput)

            if loss > 0.9999 * loss_:
                self.optimizer.lr /= 1.5

            if callback != None:
                callback(self)

            if self.optimizer.lr < 1e-5:
                break
            else:
                self.optimizer.new_epoch()

                loss_ = loss

    def to_cpu(self):
        self.model = self.model.to_cpu()
        self.xp = np

        return self

    def to_gpu(self):
        self.model = self.model.to_gpu()
        self.xp = cupy

        return self

class Model(Chain):
    def __init__(self, edim, init_hid, init_std, lindim, memsize, nhop, words):
        super(Model, self).__init__()

        normal = chainer.initializers.Normal(init_std)

        with self.init_scope():
            self.A = L.EmbedID(words, edim, normal)
            self.B = L.Linear(edim, edim, True, normal)
            self.C = L.EmbedID(words, edim, normal)
            self.T_A = L.EmbedID(memsize, edim, normal)
            self.T_C = L.EmbedID(memsize, edim, normal)
            self.W = L.Linear(edim, words, True, normal)

        self.edim = edim
        self.init_hid = init_hid
        self.lindim = lindim
        self.nhop = nhop

    def __call__(self, x):
        u = self.xp.full([x.shape[0], self.edim], self.init_hid, 'float32')
        x_ = x.reshape(-1)
        i = self.xp.arange(x_.shape[0], dtype = 'int32') % x.shape[1]
        m = F.reshape(self.A(x_) + self.T_A(i), x.shape + (self.edim,)).transpose(0, 2, 1)
        c = F.reshape(self.C(x_) + self.T_C(i), x.shape + (self.edim,)).transpose(0, 2, 1)

        for i in xrange(self.nhop):
            p = F.softmax(F.squeeze(F.batch_matmul(u, m, True)))
            o = F.squeeze(F.batch_matmul(p, c, True, True))
            u = F.split_axis(self.B(u) + o, [self.lindim], 1)
            u = F.hstack([u[0], F.relu(u[1])])

        a_hat = self.W(u)

        return a_hat

class Vocabulary(object):
    def __init__(self, corpus, words):
        self.id = {'<unk>': 0}
        self.word = {0: '<unk>'}

        for i, [j, _] in enumerate(collections.Counter(corpus).most_common(words - 1)):
            self.id[j] = i + 1
            self.word[i + 1] = j

    def id2word(self, id_):
        word = [self.word[i] if i in self.word else '<unk>' for i in id_]

        return word

    def word2id(self, word):
        id_ = [self.id[i] if i in self.id else 0 for i in word]

        return id_
