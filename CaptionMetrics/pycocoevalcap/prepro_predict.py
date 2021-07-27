import json



pred = json.load(open('/media/disk1/daic/data/saveweight/clean_topdown_box/kalphy_test_b3.json'))


my_pre={}
# for x in range(len(gt['images'])):
#     if gt['images'][x]['split']=='test':
#         tmp_id=gt['images'][x]['cocoid']
#         my_gt[tmp_id]=[]
#         for ii in range(5):
#             my_gt[tmp_id].append(gt['images'][x]['sentences'][ii]['raw'])


for x in range(len(pred)):
    tmp_id=pred[x]['image_id']
    my_pre[tmp_id] = []
    my_pre[tmp_id].append(pred[x]['caption'])

#json.dump(my_gt, open('gt_coco_test.json', 'w'))
json.dump(my_pre, open('pred_coco_topbox.json', 'w'))