import sys

sys.path.append('./pycocoevalcap/bleu')
sys.path.append('./pycocoevalcap/cider')
sys.path.append('./pycocoevalcap/meteor')
sys.path.append('./pycocoevalcap/rouge')
sys.path.append('./pycocoevalcap/spice')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json


with open('examples/ref.json', 'r') as file:
    gts = json.load(file)
with open('examples/hyp.json', 'r') as file:
    res = json.load(file)

def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    print('belu = %s' % score)

def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)

def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)

def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)
def calculate(gts,res):
    bleu(gts,res)
    cider(gts,res)
    meteor(gts,res)
    rouge(gts,res)
def main():
    gt_artery = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/GTartery.json'))
    gt_effusion = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/GTeffusion.json'))
    gt_interventricular = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/GTinterventricular.json'))
    gt_mitral = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/GTmitral.json'))
    gt_motion = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/GTmotion.json'))
    # gt_all = json.load(open(''))
    # gt_summery = json.load(open(''))

    # res_artery = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Fartery.json'))
    # res_effusion = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Feffusion.json'))
    # res_interventricular = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Finterventricular.json'))
    # res_mitral = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Fmitral.json'))
    # res_motion = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Fmotion.json'))

    # res_artery = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Nartery.json'))
    # res_effusion = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Neffusion.json'))
    # res_interventricular = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Finterventricular.json'))
    # res_mitral = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Nmitral.json'))
    # res_motion = json.load(open('/home/daic/yaxing/ImageCaptioning.pytorch/vis/eval/Nmotion.json'))
    #
    res_artery = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/Oartery.json'))
    res_effusion = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/Oeffusion.json'))
    res_interventricular = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/Ointerventricular.json'))
    res_mitral = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/Omitral.json'))
    res_motion = json.load(open('/userhome/ImageCaptioning.pytorch/vis/eval/Omotion.json'))
    # # res_all = json.load(open(''))
    # res_summery = json.load(open(''))

    # with open('examples/ref.json', 'r') as file:
    #     gts = json.load(file)
    # with open('examples/hyp.json', 'r') as file:
    #     res = json.load(file)
    print('-------artery--------')
    calculate(gt_artery, res_artery)
    print()
    print('-------effusion--------')
    calculate(gt_effusion, res_effusion)
    print()
    print('-------interventricular--------')
    calculate(gt_interventricular, res_interventricular)
    print()
    print('-------mitral--------')
    calculate(gt_mitral, res_mitral)
    print()
    print('-------motion--------')
    calculate(gt_motion, res_motion)
    print()

main()