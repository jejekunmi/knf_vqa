import json
import os
import argparse
import glob
import time
from tqdm import tqdm
import tensorflow as tf
import json
import os
from ast import literal_eval
from collections import Counter
from PIL import Image
from os.path import join as pjoin
import pickle
import util

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

from collections import Counter
import re

import nltk
from nltk.tokenize import WordPunctTokenizer
wpTokenizer = WordPunctTokenizer()

_PAD = "<pad>"
_SOS = "<sos>"
_EOS = "<eos>"
_UNK = "<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

def most_common(lst):
    return max(set(lst), key=lst.count)

def download_all(data_type, link_dict, data_dir):
    download_op = 'wget {} -P ' + data_dir
    unzip_op = 'unzip ' + data_dir + '/{} -d ' + data_dir + '/'  
    rename_op = 'mv {} ' + data_dir + '/{}'
    delete_op = 'rm -rf ' + data_dir + '/{}'

    print("\nDownloading " + data_type)
    for k, l in link_dict.items():
        outfile = "download_{}_{}".format(data_type, k)              
        
        if not os.path.exists(data_dir + '/' + outfile):
            os.system(download_op.format(l))
            os.system(unzip_op.format(l.rsplit('/', 1)[-1]))
            os.system(rename_op.format(max(glob.iglob( data_dir + '/*'), key=os.path.getctime), outfile)) 
            os.system(delete_op.format(l.rsplit('/', 1)[-1]))


def download_vqa(data_dir):
    print("\n*** Downloading datasets into {} ***".format(data_dir))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    annotations_links = {"train": "http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
                         "val"  : "http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip"}

    questions_links = {"train": "http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip",
                       "val"  : "http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                       "test" : "http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip"}
    
    images_links = { "train" : "http://msvocds.blob.core.windows.net/coco2014/train2014.zip",
                     "val"   : "http://msvocds.blob.core.windows.net/coco2014/val2014.zip",
                     "test"  : "http://msvocds.blob.core.windows.net/coco2015/test2015.zip"}
    
    comp_pairs_links = {"train": "http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
                         "val"  : "http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"}

    download_all("comp_pair_lists", comp_pairs_links, data_dir) 
    download_all("questions", questions_links, data_dir)
    download_all("annotations", annotations_links, data_dir)    
    download_all("images", images_links, data_dir) 

    print("************************************\n")  

def modify_image(image):
    # resized_images = tf.image.resize_images(images=image, size=(224, 224))
    # resized.set_shape([224,224,3])
    # flipped_images = tf.image.flip_up_down(resized)
    # return flipped_images
    print(image)
    min_dim = tf.reduce_min(tf.shape(image)[:2])
    crop_pad_image = tf.image.resize_image_with_crop_or_pad(image, min_dim, min_dim)
    resized_image = tf.image.resize_images(crop_pad_image, size=(224, 224))
    resized_image = tf.cast(resized_image, tf.uint8)
    # resized_image.set_shape((448, 448, 3))
    # print("After", resized_image)
    return resized_image

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_image(value, channels=3)
    return key,image

def inputs(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    filename,read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return filename,reshaped_image

def get_image_id(filename):
    filename = filename.decode("utf-8") 
    # print(filename)
    return int(filename[filename.rfind('_') + 1 : filename.rfind('.')])

def preprocess_image_dir(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(input_dir)

    input_files = glob.glob(input_dir + "/*")  
    #input_files = ['data/download_images_train/COCO_train2014_000000518951.jpg']
    #input_files = ['data/download_images_train/COCO_train2014_000000191320.jpg']
    '''input_files = ['data/download_images_train/COCO_train2014_000000191320.jpg',
                   'data/download_images_train/COCO_train2014_000000208754.jpg',
                   'data/download_images_train/COCO_train2014_000000236938.jpg',
                   'data/download_images_train/COCO_train2014_000000261139.jpg',
                   'data/download_images_train/COCO_train2014_000000358581.jpg',
                   'data/download_images_train/COCO_train2014_000000518951.jpg'] '''

    with tf.Graph().as_default():
        image = inputs(input_files)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(len(input_files)):
            filename,img = sess.run(image)
            #print(filename, img.shape)
            if i % 1000 == 0:
                print(output_dir, i)
            # if img.shape[-1] == 1:
            #     img = tf.image.grayscale_to_rgb(img)
            img = Image.fromarray(img, "RGB")
            img.save(os.path.join(output_dir, str(get_image_id(filename)) + ".jpeg"))

        coord.request_stop()
        coord.join(threads)





def preprocess_images(data_dir):
    print("\n*** Processing downloaded datasets ***".format(data_dir))

    preprocess_train_dir = os.path.join(data_dir, "preprocessed_images_train")
    preprocess_val_dir = os.path.join(data_dir, "preprocessed_images_val")
    preprocess_test_dir = os.path.join(data_dir, "preprocessed_images_test")


    preprocess_image_dir("data/download_images_train", preprocess_train_dir)
    preprocess_image_dir("data/download_images_val", preprocess_val_dir)
    


    print("************************************\n") 

def process_raw_anno(json_data, data_type):

    
    annotation_list = json_data["annotations"]

    print("Processing", len(annotation_list), "annotations")


    with open('data/questionid_answers_' + data_type, 'w') as f:

        for ind, annotation in enumerate(annotation_list):
            question_id = annotation["question_id"]
            answer = most_common([ans["answer"] for ans in annotation["answers"]])
            answers = [ans["answer"] for ans in annotation["answers"]]

            f.write('{}\t{}\n'.format(question_id, answers))

            if ind % 100000 == 0:
                print("Processed", ind, "answers")



def process_raw_annotations():
    print("\n*** Processing raw annotations ***")

    train_anno = json.load(open('data/download_annotations_train', 'r'))
    val_anno = json.load(open('data/download_annotations_val', 'r')) 

    process_raw_anno(train_anno, "train")
    process_raw_anno(val_anno, "val")

    print("************************************\n") 



def process_raw_ques(json_data, data_type):

    
    question_list = json_data["questions"]

    print("Processing", len(question_list), "questions")


    with open('data/questionid_imageid_question_' + data_type, 'w') as f:

        for ind, question in enumerate(question_list):
            question_id = question["question_id"]
            
            f.write('{}\t{}\t{}\n'.format(question["question_id"], 
                                          question["image_id"],
                                          question["question"]))

            if ind % 100000 == 0:
                print("Processed", ind, "answers")



def process_raw_questions():
    print("\n*** Processing raw questions ***")

    train_questions = json.load(open('data/download_questions_train', 'r'))
    val_questions = json.load(open('data/download_questions_val', 'r')) 

    process_raw_ques(train_questions, "train")
    process_raw_ques(val_questions, "val")

    print("************************************\n") 


def combine_raw_join(data_type):
    with open('data/combined_raw_' + data_type, 'w') as f:
        with open("data/questionid_imageid_question_" + data_type) as f1, open('data/questionid_answers_' + data_type) as f2:
            for qid_iid_q, qid_a in zip(f1, f2):
                question_id1, image_id, question = qid_iid_q.strip().split('\t')
                question_id2, answers = qid_a.strip().split('\t')

                assert(question_id1 == question_id2)

                f.write("{}\t{}\t{}\t{}\n".format(question_id1, image_id, question, answers))

def combine_raw():
    print("\n*** Combining raw data ***")

    combine_raw_join("train")
    combine_raw_join("val")

    print("************************************\n")     

def normalize(word):
    return word.lower()

def tokenize(sequence):
    sequence = normalize(sequence)
    first_tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    tokens = []
    for t in first_tokens:
        if "/" in t:
            tokens.extend(wpTokenizer.tokenize(t))
        else:
            tokens.append(t)
    return tokens

def add_to_vocab(vocab, sentence):
    sentence = normalize(sentence)
    tokens = tokenize(sentence)
    vocab.update(tokens)

def add_to_vocab_from_file(vocab, file_name):
    c = 1
    with open(file_name) as f_combined:
        for line in f_combined:
            # print(line)
            question_id, image_id, question, answers = line.strip().split('\t')
            answers = literal_eval(answers)

            add_to_vocab(vocab, question)
            for ans in answers:
                add_to_vocab(vocab, ans)
            
            if c % 10000 == 0:
                print(c, file_name, len(vocab))
                # break
                
            c += 1

def build_vocab():
    print("\n*** Building vocab ***")
    vocab = Counter()

    add_to_vocab_from_file(vocab, "data/combined_raw_train")
    add_to_vocab_from_file(vocab, "data/combined_raw_val")

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True) 
    
    with open("data/vocab.dat", "w") as vocab_file:
        for w in vocab_list:
            vocab_file.write("{}\n".format(w))
    # print(len(vocab))

    print("************************************\n")


def create_token_file(file_name):
    with open(file_name) as f_combined:
        with open(file_name[:file_name.rfind("_")] + "_tokens_" + file_name[file_name.rfind("_")+1:] , 'w') as f_tokens:
            c = 1
            for line in f_combined:
                # print(line)
                question_id, image_id, question, answers = line.strip().split('\t')
                question_tokens = tokenize(question)

                answers = literal_eval(answers)
                answers_tokens = []
                for ans in answers:
                    answers_tokens.append(tokenize(ans))

                f_tokens.write("{}\t{}\t{}\t{}\n".format(question_id, image_id, question_tokens, answers_tokens))
                
                if c % 50000 == 0:
                    print(c, file_name)
                    #break
                    
                c += 1

def data_with_tokens():
    print("\n*** Creating token data ***")
    
    create_token_file("data/combined_raw_train")
    create_token_file("data/combined_raw_val")

    print("************************************\n")

def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    rev_vocab = []
    with open(vocabulary_path) as f:
        rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip('\n') for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab


def create_ids_file(file_name, vocab):
    with open(file_name) as f_tokens:
        with open("data/ids_" + file_name[file_name.rfind("_")+1:] , 'w') as f_ids:
            c = 1
            for line in f_tokens:
                # print(line)
                question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

                # print(question_tokens, answers_tokens)
                
                question_tokens = literal_eval(question_tokens)
                answers_tokens = literal_eval(answers_tokens)

                # print(question_tokens, answers_tokens)

                question_as_ids = [vocab[q] for q in question_tokens]

                answers_as_ids = []
                for ans in answers_tokens:
                    answers_as_ids.append([vocab[a] for a in ans])

                # print(question_tokens, answers_tokens)

                f_ids.write("{}\t{}\t{}\t{}\n".format(question_id, image_id, question_as_ids, answers_as_ids))
                
                if c % 50000 == 0:
                    print(c, file_name)
                    # break
                    
                c += 1

def data_with_ids():
    print("\n*** Creating ids data ***")
    
    vocab, rev_vocab = initialize_vocabulary("data/vocab.dat")

    create_ids_file("data/combined_raw_tokens_train", vocab)
    create_ids_file("data/combined_raw_tokens_val", vocab)

    print("************************************\n")


def process_glove(size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    glove_dim = 100
    _, vocab_list = initialize_vocabulary(pjoin("data", "vocab.dat"))
    save_path = "data/glove.trimmed.{}".format(glove_dim)

    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join("data/dwr/", "glove.6B.{}d.txt".format(glove_dim))
        glove = np.zeros((len(vocab_list), glove_dim))
        not_found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                elif word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                elif word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                elif word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                else:
                    not_found += 1
        found = size - not_found
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def token_histogram_file(filename, question_lengths, answer_lengths):
    with open(filename) as f:
        c = 1
        for line in f:
            question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

            question_tokens = literal_eval(question_tokens)
            answers_tokens = literal_eval(answers_tokens)

            question_lengths.append(len(question_tokens))

            for ans in answers_tokens:
                answer_lengths.append(len(ans))

            if c % 100000 == 0:
                print(c, filename)
            c += 1



def token_histogram():
    question_lengths = [] 
    answer_lengths = []

    token_histogram_file("data/ids_train", question_lengths, answer_lengths)
    token_histogram_file("data/ids_val", question_lengths, answer_lengths)    

    plotter.hist(question_lengths, bins=100)
    plotter.title("Question Length Histogram")
    plotter.xlabel("Question length")
    plotter.ylabel("Frequency")
    plotter.savefig("question.png")

    plotter.hist(answer_lengths, bins=100)
    plotter.title("Answer Length Histogram")
    plotter.xlabel("Answer length")
    plotter.ylabel("Frequency")
    plotter.savefig("answer.png")

def token_max_file(filename, question_lengths, answer_lengths):

    with open(filename) as f:
        c = 1
        for line in f:
            question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

            question_tokens = literal_eval(question_tokens)
            answers_tokens = literal_eval(answers_tokens)

            question_lengths.append(len(question_tokens))
            if len(question_tokens) > 20:
                print(question_id)

            for ans in answers_tokens:
                answer_lengths.append(len(ans))
                # if len(ans) > 15:
                #     print(question_id)

            if c % 100000 == 0:
                print(c, filename)
            c += 1

def token_max():
    question_lengths = [] 
    answer_lengths = []

    token_max_file("data/ids_train", question_lengths, answer_lengths)
    token_max_file("data/ids_val", question_lengths, answer_lengths)    

    print("Max Question length:", max(question_lengths))
    print("Max Answer length:", max(answer_lengths))

def lines_to_padded_np_array(data, max_length):
    padded_seqs = []
    masks = []
    
    for sentence in data:
        sentence_len = len(sentence)
        sentence = [int(s) for s in sentence]
        if sentence_len >= max_length:
            padded_seqs.append(np.array(sentence[:max_length]))
            masks.append(max_length)
        else:
            p_len = max_length - sentence_len
            new_sentence = sentence + [0] * p_len
            padded_seqs.append(np.array(new_sentence))
            masks.append(sentence_len)
    return {'data': np.array(padded_seqs), 'mask': np.array(masks)}

def toy_train_data():
    all_questions = []
    all_answers = []

    with open("data/ids_train") as f:
        c = 1
        for line in f:
            question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

            question_tokens = literal_eval(question_tokens)
            answers_tokens = literal_eval(answers_tokens)

            all_questions.append(question_tokens)

            a = most_common([ans[0] for ans in answers_tokens])
            # print(a)
            all_answers.append(a)                

            if c % 100000 == 0:
                print(c)
            c += 1

    padded = lines_to_padded_np_array(all_questions, 20)
    all_questions = padded['data']
    questions_mask = padded['mask']

    all_answers = np.array(all_answers)

    np.savez("data/toy_train", questions=all_questions, mask=questions_mask, answers=all_answers)


def toy_val_data():
    all_questions = []
    all_answers = []

    with open("data/ids_val") as f:
        c = 1
        for line in f:
            question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

            question_tokens = literal_eval(question_tokens)
            answers_tokens = literal_eval(answers_tokens)

            all_questions.append(question_tokens)

            # a = most_common([ans[0] for ans in answers_tokens])
            # print(a)
            all_answers.append(answers_tokens)                

            if c % 100000 == 0:
                print(c)
            c += 1

    padded = lines_to_padded_np_array(all_questions, 20)
    all_questions = padded['data']
    questions_mask = padded['mask']

    

    np.savez("data/toy_val_questions", questions=all_questions, mask=questions_mask)

    with open('data/toy_val_answers', 'wb') as fp:
        pickle.dump(all_answers, fp)

def create_clean_data_for_type(data_type):
    all_questions = []
    all_answers = []
    all_question_ids = []
    all_image_ids = []

    with open("data/ids_" + data_type) as f:
        c = 1
        for line in f:
            question_id, image_id, question_tokens, answers_tokens = line.strip().split('\t')

            question_tokens = literal_eval(question_tokens)
            answers_tokens = literal_eval(answers_tokens)

            all_question_ids.append(question_id)
            all_image_ids.append(image_id)
            all_questions.append(question_tokens)

            # a = most_common([ans[0] for ans in answers_tokens])
            # print(a)
            all_answers.append(answers_tokens)                

            if c % 50000 == 0:
                print(c)
                # break
            c += 1

    padded = lines_to_padded_np_array(all_questions, 25)
    
    all_question_ids = np.array(all_question_ids)
    all_image_ids = np.array(all_image_ids)
    all_questions = padded['data']
    all_questions_mask = padded['mask'] # (a != 0).sum(1)
    


    np.savez("data/questions_" + data_type, questions=all_questions)
    np.savez("data/question_ids_" + data_type, question_ids=all_question_ids)
    np.savez("data/image_ids_" + data_type, image_ids=all_image_ids)


    with open('data/answers_' + data_type, 'wb') as fp:
        pickle.dump(all_answers, fp)

def create_clean_data():
    create_clean_data_for_type("train")
    create_clean_data_for_type("val")

def add_to_classes_from_file(classes, file_name):
    c = 1
    with open(file_name) as f_combined:
        for line in f_combined:
            # print(line)
            question_id, image_id, question, answers = line.strip().split('\t')
            answers = literal_eval(answers)

            for ans in answers:
                classes[normalize(ans)] += 1
            
            if c % 10000 == 0:
                print(c, file_name, len(classes ))
                # break
                
            c += 1
    

def build_classes():
    print("\n*** Building classes ***")
    classes = Counter()

    add_to_classes_from_file(classes, "data/combined_raw_train")
    add_to_classes_from_file(classes, "data/combined_raw_val")

    class_list = sorted(classes.items(), key=lambda x: -x[1])
    # class_list = sorted(classes, key=classes.get, reverse=True) 
    
    with open("data/classes.dat", "w") as class_file:
        for w in class_list:
            class_file.write("{}\t{}\n".format(w[1], w[0]))
    print(len(class_list))

    print("************************************\n")

def build_answers_from_file(answer_to_id, file_name, data_type):
    all_answers = []
    c = 0
    with open(file_name) as f_combined:
        for line in f_combined:
            answers_for_question = []
            question_id, image_id, question, answers = line.strip().split('\t')
            answers = literal_eval(answers)

            for ans in answers:
                answers_for_question.append(answer_to_id[normalize(ans)])
            all_answers.append(answers_for_question)
            if c % 100000 == 0:
                print(c, file_name)
                # break
                
            c += 1
    np.savez("data/answers_" + data_type, answers=np.array(all_answers))
    
def build_answers():
    print("\n*** Building classes ***")
    answer_to_id, id_to_answer = util.load_answer_map(min_count=-1)

    build_answers_from_file(answer_to_id, "data/combined_raw_train", "train")
    build_answers_from_file(answer_to_id, "data/combined_raw_val", "val")

    print("************************************\n")
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--download', default='False', help='Download and extract data from VQA server')
    parser.add_argument('--dir', default='data', help='Download and extract data from VQA server')
    parser.add_argument('--preprocess_images', default='False', help='Process downloaded data')
    parser.add_argument('--process_raw_annotations', default='False', help='Process downloaded data')
    parser.add_argument('--process_raw_questions', default='False', help='Process downloaded data')
    parser.add_argument('--combine_raw', default='False', help='Process downloaded data')
    parser.add_argument('--build_vocab', default='False', help='Process downloaded data')
    parser.add_argument('--data_with_tokens', default='False', help='Process downloaded data')
    parser.add_argument('--data_with_ids', default='False', help='Process downloaded data')
    parser.add_argument('--process_glove', default='False', help='Process downloaded data')
    parser.add_argument('--token_histogram', default='False', help='Process downloaded data')
    parser.add_argument('--toy_train_data', default='False', help='Process downloaded data')
    parser.add_argument('--toy_val_data', default='False', help='Process downloaded data')
    parser.add_argument('--create_clean_data', default='False', help='Process downloaded data')
    parser.add_argument('--build_classes', default='False', help='Process downloaded data')

    args = parser.parse_args()
    params = vars(args)
    print('Parsed input parameters:')
    print(json.dumps(params, indent = 2))
    data_prefix = os.path.join(params['dir'])

    if params['download'] == 'True':
        download_vqa(data_prefix)

    if params['preprocess_images'] == 'True':
        preprocess_images(data_prefix)

    if params['process_raw_annotations'] == 'True':
        process_raw_annotations()

    if params['process_raw_questions'] == 'True':
        process_raw_questions()

    if params['combine_raw'] == 'True':
        combine_raw()

    if params['build_vocab'] == 'True':
        build_vocab() 

    if params['data_with_tokens'] == 'True':
        data_with_tokens() 

    if params['data_with_ids'] == 'True':
        data_with_ids() 

    if params['process_glove'] == 'True':
        process_glove() 

    if params['token_histogram'] == 'True':
        token_histogram() 

    if params['toy_train_data'] == 'True':
        toy_train_data() 

    if params['toy_val_data'] == 'True':
        toy_val_data() 

    if params['create_clean_data'] == 'True':
        create_clean_data() 

    if params['build_classes'] == 'True':
        build_classes() 