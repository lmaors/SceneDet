# coding: utf-8

from __future__ import division, print_function

from nltk.tokenize import word_tokenize
import nltk
import argparse

import models
import data
import os.path as osp
import theano
import sys
import re
from io import open

import theano.tensor as T
import numpy as np

numbers = re.compile(r'\d')
is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return ' '
    elif punct_token.startswith('-'):
        return ' ' + punct_token[0] + ' '
    else:
        return punct_token[0] + ' '

def punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, show_unk):
    if len(words) == 0:
        sys.exit("Input text from stdin missing.")

    if words[-1] != data.END:
        words += [data.END]

    i = 0
    while True:

        subsequence = words[i:i+data.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(
                "<NUM>" if is_number(w) else w.lower(),
                word_vocabulary[data.UNK])
            for w in subsequence]

        if show_unk:
            subsequence = [reverse_word_vocabulary[w] for w in converted_subsequence]

        y = predict(to_array(converted_subsequence))

        f_out.write(subsequence[0].title())

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(y_t.flatten())
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            current_punctuation = punctuations[j]
            f_out.write(convert_punctuation_to_readable(current_punctuation))
            if j < step - 1:
                if current_punctuation in data.EOS_TOKENS:
                    f_out.write(subsequence[1+j].title())
                else:
                    f_out.write(subsequence[1+j])

        if subsequence[-1] == data.END:
            break

        i += step

    f_out.seek(0)
    return  f_out.read()

def reduce_punctuator(text):
    parser = argparse.ArgumentParser('correct punctuator ')
    parser.add_argument('--model_path',default='../pretrained_models/Demo-Europarl-EN.pcl')
    parser.add_argument('--show_unk',default=False)
    args = parser.parse_args()
    x = T.imatrix('x')

    print("Loading model parameters...")
    net, _ = models.load(args.model_path, 1, x)

    print("Building model...")

    predict = theano.function(inputs=[x], outputs=net.y)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_word_vocabulary = {v:k for k,v in net.x_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

    human_readable_punctuation_vocabulary = [p[0] for p in punctuation_vocabulary if p != data.SPACE]
    tokenizer = word_tokenize
    untokenizer = lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    with open('tmp.txt','w+') as f_out:
        words = [w for w in untokenizer(' '.join(tokenizer(text))).split()
                 if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]

        return punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, args.show_unk)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('correct punctuator ')
    parser.add_argument('--model_path',default='../pretrained_models/Demo-Europarl-EN.pcl')
    parser.add_argument('--show_unk',default=False)
    args = parser.parse_args()



    a = reduce_punctuator('firstly development policy in africa in any case in the acp countries employment policy in madeira the canaries guadeloupe martinique and crete regional policy in the ultra-peripheral areas human rights which mr barthet-mayer mentioned earlier since dollar bananas are after all slavery bananas the product of human exploitation by three multinationals payments of ecu 50 per month instead of ecu 50 per day in guadeloupe or martinique it also brings into question budgetary policy because the european union is after all making a present of ecu 1.9 billion to three multinationals where are the financial interests of the european union')
    print(a)
    # with open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False) as f_out:
    #     while True:
    #         try:
    #             text = input("\nTEXT: ")
    #         except NameError:
    #             text = input("\nTEXT: ")
    #
    #         words = [w for w in untokenizer(' '.join(tokenizer(text))).split()
    #                  if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]
    #
    #         punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, args.show_unk)
    #         f_out.flush()
