import json
import random
import numpy as np
import dynet as dy
import config
import time
import matplotlib.pyplot as plt
start_time = time.time()
LABELS = {
    u"entailment": 0,
    u"contradiction": 1,
    u"neutral": 2
}

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)


class Embedding:
    def __init__(self, file_name, words_in_file):
        self.words_ix = {}
        self.words = self.build_vectors(file_name, words_in_file)
        self.UNK = self.unk_vector()
        self.words["UNK"] = self.UNK
        self.words_ix["UNK"] = len(self.words_ix)
        self.words_to_ix = { id : word for word, id in self.words_ix.iteritems() }
        # self.known = 0
        # self.unknown = 0

    def get_embed_word(self, word):
        if word in self.words:
            # self.known += 1
            return self.words_ix[word]
        # self.unknown += 1
        return self.words_ix["UNK"]

    def get_embed_from_ix(self, id):
        return self.words[self.words_to_ix[id]]

    def all_embeds_from_ix(self, ids):
        return np.array(map(self.get_embed_from_ix, ids))

    def all_embed_words(self, words):
        return np.array(map(self.get_embed_word, words))

    def unk_vector(self):
        vectors = []
        if len(self.words) == 0:
            raise ValueError("Build the words vectors before creating the UNK!")
        for i, word in enumerate(self.words):
            if i > 100:
                break
            vectors.append(self.words[word])
        return np.average(vectors, axis=0)

    def build_vectors(self, file_name, words_in_file):
        words = {}
        print "Start reading the embedding file " + file_name
        # lines = open(file_name, "r").read().split("\n")
        with open(file_name, "r") as lines:
            for line in lines:
                vector = line.split()
                word = vector.pop(0)
                if word in words_in_file:
                    words[word] = np.array(map(float, vector))
                    norm=np.linalg.norm(words[word], ord=2)

                    words[word] = words[word]/norm
                    self.words_ix[word] = len(self.words_ix)
        print "Finish reading the embedding " + str(passed_time(start_time))
        return words


class SNLI:
    def __init__(self, type, file_name):
        self.data = []
        print "Start reading the snli type " + type + " in file " + file_name
        self.sentence = []
        self.words = {}
        with open(file_name, "r") as lines:
            for line in lines:
                data = json.loads(line)
                gold = data["gold_label"]
                if gold != u"-":
                    label = LABELS[gold]
                    sent1 = data["sentence1"].lower().rstrip(".").split()
                    sent2 = data["sentence2"].lower().rstrip(".").split()
                    if len(sent1) <= 10 and len(sent2) <= 10:
                        for word in sent1:
                            if word not in self.words:
                                self.words[word] = len(self.words)
                        for word in sent2:
                            if word not in self.words:
                                self.words[word] = len(self.words)

                        self.sentence.append((sent1, sent2, label))
                # self.sentence.append((data["sentence1"].lower().rstrip(".").split(), data["sentence2"].lower().rstrip(".").split(), label))
                    # if len(sent1) <= 10 and len(sent2) <= 10:
                    #     sentence1 = embedding.all_embed_words(sent1)
                    #     sentence2 = embedding.all_embed_words(sent2)
                    #     # # print sentence1
                    #     #
                    #     self.data.append((sentence1, sentence2, label))
                    # gfd
        print "Got from the file " + file_name + " " + str(len(self.data)) + " pairs"

    def define_sent_with_embeddings(self, embedding):
        for (sent1, sent2, label) in self.sentence:
            sentence1 = embedding.all_embed_words(sent1)
            sentence2 = embedding.all_embed_words(sent2)
            self.data.append((sentence1, sentence2, label))

class FeedForward(object):
    def __init__(self, model, size_w_1, size_w_2, drop_param):
        pc = model.add_subcollection()
        self.W_1 = pc.add_lookup_parameters(size_w_1)
        self.W_2 = pc.add_lookup_parameters(size_w_2)
        self.pc = pc
        self.drop_param = drop_param
    def __call__(self, sentence1, sentence2):
        W_1 = dy.parameter(self.W_1)
        # relu activation with dropout
        out1 = dy.rectify(dy.dropout(sentence1, self.drop_param) * W_1)
        out2 = dy.rectify(dy.dropout(sentence2, self.drop_param) * W_1)

        W_2 = dy.parameter(self.W_2)
        out1 = dy.rectify(dy.dropout(out1, self.drop_param) * W_2)
        out2 = dy.rectify(dy.dropout(out2, self.drop_param) * W_2)
        return out1, out2

class Trainer:
    def __init__(self, embedding_size, hidden_size, labels_size, embedding):
        self.embedding = embedding
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        # self.trainer = dy.AdagradTrainer(self.model, 0.05)

        self.linear = self.model.add_parameters((embedding_size, hidden_size))
        # print hidden_size
        self.feed_F = FeedForward(self.model, (hidden_size, hidden_size), (hidden_size, hidden_size), 0.2)
        self.feed_G = FeedForward(self.model, (hidden_size, 2*hidden_size), (hidden_size, hidden_size), 0.2)

        # self.mlpG1 = self.model.add_parameters((2 * hidden_size, hidden_size))
        # self.mlpG2 = self.model.add_parameters((hidden_size, hidden_size))

        self.h_step_1 = self.model.add_parameters((2*hidden_size, hidden_size))
        self.h_step_2 = self.model.add_parameters((hidden_size, hidden_size))

        self.linear2 = self.model.add_parameters((hidden_size, labels_size))

    def predict(self, s1, s2):
        return self.apply(s1, s2)

    def accuracy(self, data):
        good = 0.0
        bad = 0.0
        for s1, s2, label in data:
            prob = self.apply(s1, s2)
            softmax = dy.softmax(prob).npvalue()
            pred = np.argmax(softmax)
            if pred == label:
                good += 1
            else:
                bad += 1
        return good / (good+bad)

    def __call__(self, sent1, sent2):
        return self.apply(sent1, sent2)

    def apply(self, sent1, sent2):
        eL = dy.parameter(self.linear)
        sent1 = dy.inputTensor(self.embedding.all_embeds_from_ix(sent1)) * eL
        sent2 = dy.inputTensor(self.embedding.all_embeds_from_ix(sent2)) * eL

        out1, out2 = self.feed_F(sent1, sent2)
        e_out = out1*dy.transpose(out2)
        prob_f_1 = dy.softmax(e_out)
        score = dy.transpose(e_out)
        prob_f_2 = dy.softmax(score)

        sent1_allign = dy.concatenate_cols([sent1, prob_f_1 * sent2])
        sent2_allign = dy.concatenate_cols([sent2, prob_f_2 * sent1])

        out_g_1, out_g_2 = self.feed_G(sent1_allign, sent2_allign)

        sent1_out_g = dy.sum_dim(out_g_1, [0])
        sent2_out_g = dy.sum_dim(out_g_2, [0])

        concat = dy.transpose(dy.concatenate([sent1_out_g, sent2_out_g]))

        h_step_1 = dy.parameter(self.h_step_1)
        sent_h = dy.rectify(dy.dropout(concat, 0.2) * h_step_1)
        h_step_2 = dy.parameter(self.h_step_2)
        sent_h = dy.rectify(dy.dropout(sent_h, 0.2) * h_step_2)

        final = dy.parameter(self.linear2)
        final = dy.transpose(sent_h * final)
        return final

def save_in_graph(data, type):
    plt.figure(0)
    plt.plot(range(len(data)), [a[i] for a in data])
    plt.xlabel('Epochs')
    plt.ylabel(type)
    plt.savefig(type + '.png')

if __name__ == '__main__':
    print "Load train"
    train = SNLI("train", config.SNLI_ROOT + "train.jsonl")
    embedding = Embedding(config.GLOVE_ROOT + "glove.42B.300d.txt", train.words)
    train.define_sent_with_embeddings(embedding)
    # load train/dev/test data
    print "="*20
    print "Load dev"
    # dev = SNLI("dev", config.SNLI_ROOT + "dev.jsonl", embedding)
    print "="*20
    print "Load test"
    test = SNLI("test", config.SNLI_ROOT + "test.jsonl")
    test.define_sent_with_embeddings(embedding)

    model = Trainer(300, 200, len(LABELS), embedding)

    # modelFileCache = Path(args.model)
    # if modelFileCache.is_file():
    #     model.load(args.model)

    losses = []

    loss = 0
    checked = 0
    test_accuracy = []
    train_accuracy = []
    batch_size = 32
    for epoch in range(3):
        print "Start epoch " + str(epoch + 1)
        random.shuffle(train.data)
        dy.renew_cg()

        good = 0.0
        bad = 0.0
        errors = []
        # batching
        for i, (s1, s2, label) in enumerate(train.data):
            prob = model(s1, s2)
            softmax = dy.softmax(prob).npvalue()
            pred = np.argmax(softmax)
            error = dy.pickneglogsoftmax(prob, label)
            errors.append(error)
            if pred == label:
                good += 1
            else:
                bad += 1
            if i % batch_size == 0 and i > 0:
                # print len(errors)
                # print errors
                sum_errors = dy.esum(errors)
                loss += sum_errors.value()
                sum_errors.backward()
                model.trainer.update()
                checked += batch_size
                dy.renew_cg()
                errors = []


                # errors = []
                # dy.renew_cg()
                # if i > 100*20:
                #     print "done"
                #     print i
                # finish the batch
            if i % (batch_size * 5) == 0 and i > 0:
                avgLoss = loss / checked
                losses.append(avgLoss)
                print "-"*20
                print "Time: " + str(passed_time(start_time))
                print  "Epoch: " + str(epoch+1) + ", Iteration: " + str(i) + " Average loss: " + str(avgLoss)
                loss = 0
                checked = 0
                print "Start calculation of accuracy on test, " + str(len(test.data)) + " examples"
                accuracy = model.accuracy(test.data)
                test_accuracy.append(accuracy)
                print "Test Accuracy: " + str(accuracy)
                print "Time passed " + str(passed_time(start_time))
                # print "Start calculation of accuracy on train, " + str(len(train.data)) + " examples"
                # accuracy_train = model.accuracy(train.data)
                train_accuracy.append(good/(good+bad))
                print "Train Accuracy: " + str(good/(good+bad))
                good = 0.0
                bad = 0.0

                # model.save(args.model)


    save_in_graph(losses, "loss_train")
    save_in_graph(test_accuracy, "test_accuracy")
    save_in_graph(train_accuracy, "train_accuracy")
