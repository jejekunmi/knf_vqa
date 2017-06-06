import os
import numpy as np
from os.path import join as pjoin
import pickle
from collections import Counter
import time
import sys
import tensorflow as tf

_UNK = "<unk>"

def initialize_model(session, model, model_dir, train_saved_model, config):
    
    if train_saved_model:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        model.encoder.vgg_net.load_weights(weight_file=config.vgg16_weight_file, sess=session)
        print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def get_batch_from_indices(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def get_minibatches_indices(indices, minibatch_size, shuffle=True):
    data_size = len(indices)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        batch_indices.append(indices[minibatch_start:minibatch_start + minibatch_size]) 
    return batch_indices

def load_answer_map(min_count=20):
    id_to_answer = [_UNK]    
    with open("data/classes.dat") as class_file:
        for line in class_file:
            count, word = line.strip().split("\t")
            if int(count) >= min_count:
                id_to_answer.append(word)
    answer_to_id = dict([(x, y) for (y, x) in enumerate(id_to_answer)])
    return answer_to_id, id_to_answer

def load_data(data_type, min_count):
    questions = np.load("data/questions_" + data_type + ".npz")["questions"]
    questions_mask = (questions != 0).sum(1)
    question_ids = np.load("data/question_ids_" + data_type + ".npz")["question_ids"]
    image_ids = np.load("data/image_ids_" + data_type + ".npz")["image_ids"]
    answers = np.load("data/answers_" + data_type + ".npz")["answers"]
    answer_to_id, id_to_answer = load_answer_map(min_count)
    
    # Make uncommon words unknown
    answers[answers >= len(answer_to_id)] = 0
    
    return {"questions": questions,
            "questions_mask": questions_mask,
            "question_ids": question_ids,
            "image_ids": image_ids,
            "answers": answers}

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def load_data_all(train_min_count=20, val_cutoff=107183, limit=10000000,
                  filter_unknown=True):
    all_data = {"train": load_data("train", min_count=train_min_count),
                "val": load_data("val",min_count=train_min_count)} 

    Questions_train = all_data["train"]["questions"]
    Questions_mask_train = all_data["train"]["questions_mask"]
    Image_ids_train = all_data["train"]["image_ids"]
    All_answers_train =  all_data["train"]["answers"]
    Answers_train = most_common_answers(All_answers_train)
    if filter_unknown:
        known = (Answers_train != 0)
        Questions_train = Questions_train[known]
        Questions_mask_train = Questions_mask_train[known]
        Image_ids_train = Image_ids_train[known]
        All_answers_train =  All_answers_train[known]
        Answers_train = Answers_train[known]

    Questions_val = all_data["val"]["questions"][:val_cutoff]
    Questions_mask_val = all_data["val"]["questions_mask"][:val_cutoff]
    Image_ids_val = all_data["val"]["image_ids"][:val_cutoff]
    All_answers_val = all_data["val"]["answers"][:val_cutoff]
    Answers_val = most_common_answers(All_answers_val)
    if filter_unknown:
        known = (Answers_val != 0)
        Questions_val = Questions_val[known]
        Questions_mask_val = Questions_mask_val[known]
        Image_ids_val = Image_ids_val[known]
        All_answers_val =  All_answers_val[known]
        Answers_val = Answers_val[known]

    Questions_test = all_data["val"]["questions"][val_cutoff+1:]
    Questions_mask_test = all_data["val"]["questions_mask"][val_cutoff+1:]
    Image_ids_test = all_data["val"]["image_ids"][val_cutoff+1:]
    All_answers_test = all_data["val"]["answers"][val_cutoff+1:]
    Answers_test = most_common_answers(All_answers_test)
    if filter_unknown:
        known = (Answers_test != 0)
        Questions_test = Questions_test[known]
        Questions_mask_test = Questions_mask_test[known]
        Image_ids_test = Image_ids_test[known]
        All_answers_test =  All_answers_test[known]
        Answers_test = Answers_test[known]

    dataset = {"train": {"questions": Questions_train[:limit], 
                         "mask":      Questions_mask_train[:limit],
                         "image_ids": Image_ids_train[:limit],
                         "all_answers": All_answers_train[:limit],
                         "answers":   Answers_train[:limit]},
               "val"  : {"questions": Questions_val[:limit], 
                         "mask":      Questions_mask_val[:limit],
                         "image_ids": Image_ids_val[:limit],
                         "all_answers":   All_answers_val[:limit],
                         "answers":   Answers_val[:limit]},
               "test" : {"questions": Questions_test[:limit], 
                         "mask":      Questions_mask_test[:limit],
                         "image_ids": Image_ids_test[:limit],
                         "all_answers":   All_answers_test[:limit],
                         "answers":   Answers_test[:limit]}}
    
    return obj(dataset)
    
def most_common_answers(all_answers):
    most_common_answers = []
    for ans in all_answers:
        c = Counter(ans)
        most_common_answers.append(c.most_common(1)[0][0])
    return np.array(most_common_answers)

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)