#!/usr/bin/env python
import sys
import os
import os.path
import nltk

import math
from fractions import Fraction
import warnings
from collections import Counter
from nltk.util import ngrams

# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import single_meteor_score

from utils import sentence_bleu, single_meteor_score

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    truth_file = os.path.join(truth_dir, "truth.txt")
    truth = open(truth_file, 'r')
    truth = truth.readlines()
    
    submission_answer_file = os.path.join(submit_dir, "answer.txt")
    submission_answer = open(submission_answer_file, 'r')
    submission_answer = submission_answer.readlines()
    print(truth)
    total_num = len(truth)
    total_bleu_scores = 0
    total_meteor_scores = 0
    for i in range(total_num):
        total_bleu_scores += sentence_bleu([truth[i].split(" ")], submission_answer[i].split(" "))
        total_meteor_scores += single_meteor_score(truth[i], submission_answer[i])

    bleu_result = total_bleu_scores/total_num
    meteor_result = total_meteor_scores/total_num

    output_file.write('bleu_score: ' + str(bleu_result))
    output_file.write('\n')
    output_file.write('meteor: ' + str(meteor_result))
    output_file.close()



