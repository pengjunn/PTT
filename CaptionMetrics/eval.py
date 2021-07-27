import os
import sys
import json
import argparse
import random
import numpy as np
from ipdb import set_trace

sys.path.append('./pycocoevalcap/bleu')
sys.path.append('./pycocoevalcap/cider')
# sys.path.append('./pycocoevalcap/meteor')
sys.path.append('./pycocoevalcap/rouge')
# sys.path.append('./pycocoevalcap/spice')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.spice.spice import Spice


def bleu(gts, res):
    scorer = Bleu(n=4)
    score, scores = scorer.compute_score(gts, res)
    # print('bleu = %s' % score)
    return score


def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    # print('cider = %s' % score)
    return score


# def meteor(gts, res):
#     scorer = Meteor()
#     # set_trace()
#     score, scores = scorer.compute_score(gts, res)
#     # print('meter = %s' % score)
#     return score


def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    # print('rouge = %s' % score)
    return score


# def spice(gts, res):
#     scorer = Spice()
#     score, scores = scorer.compute_score(gts, res)
#     # print('spice = %s' % score)
#     return score


def eval_results(gts, res):
    b_score = bleu(gts, res)
    c_score = cider(gts, res)
    r_score = rouge(gts, res)

    return [b_score, c_score, r_score]


def get_tmp_gts(gts, res):
    # set_trace()
    tmp_gts = {}
    for k in res.keys():
        if '_s-1' in k:
            gts_k = k.split('_s-1')[0]
        else:
            gts_k = os.path.splitext(k)[0]
        try:
            tmp_gts[k] = gts[gts_k]
        except:
            res.pop(k)

    return tmp_gts


def get_tmp_gts_new(gts, res):
    # for eval 3k 
    # set_trace()
    tmp_gts = {}
    for k in res.keys():
        if '_s-1' in k:
            gts_k = k.split('_s-1')[0]
        else:
            gts_k = os.path.splitext(k)[0]
            gts_k = '_'.join(gts_k.split('_')[1:])
        try:
            tmp_gts[k] = gts[gts_k]
        except:
            res.pop(k)

    return tmp_gts


def eval_json(gts_json_file, res_json_file):
    with open(gts_json_file, 'r') as f:
        gts = json.load(f)
    with open(res_json_file, 'r') as f:
        res = json.load(f)
    
    res_key_list = res.keys()
    assert args.num_images <= len(res_key_list), 'error num_images'
    
    if (args.num_images == len(res_key_list)) or (args.num_images == -1):
        tmp_res = res
    else:
        tmp_keys = random.sample(res_key_list, args.num_images)
        tmp_res = {}
        for key in tmp_keys:
            tmp_res[key] = res[key]
    
    tmp_gts = get_tmp_gts_new(gts, tmp_res)
    eval_res = eval_results(tmp_gts, tmp_res)
    return eval_res


def preview(bs, c_score, r_score):
    print 'B: [%.5f, %.5f, %.5f, %.5f]' % (bs[0], bs[1], bs[2], bs[3])
    print 'C:  %.5f' % (c_score)
    print 'R:  %.5f' % (r_score)

if __name__ == '__main__':
    """
        usage:
        python eval.py --gts_file path/to/gt.json --res_file path/to/sample.json
    """
    
    # Input arguments and options
    parser = argparse.ArgumentParser()
    # For evaluation on a folder of images:
    parser.add_argument('--gts_file', type=str, 
                        default='examples/coco_val_gts.json',
                        help='')
    parser.add_argument('--res_file', type=str, 
                        default='',
                        help='')
    parser.add_argument('--num_images', type=int, default=-1, help='')
    parser.add_argument('--iter', type=int, default=1, help='')
    args = parser.parse_args()

    gts = args.gts_file
    res = args.res_file
    B1, B2, B3, B4 = 0, 0, 0, 0
    C, R = 0, 0, 0
    for i in range(args.iter):
        b_scores, c_score, r_score = eval_json(gts, res)
        # preview(b_scores, c_score, m_score, r_score)
        B1 += b_scores[0]
        B2 += b_scores[1]
        B3 += b_scores[2]
        B4 += b_scores[3]
        C += c_score
        R += r_score

    it = float(args.iter)
    print 'B: [%.5f, %.5f, %.5f, %.5f]' % (B1/it, B2/it, B3/it, B4/it)
    print 'C:  %.5f' % (C / it)
    print 'R:  %.5f' % (R / it)
