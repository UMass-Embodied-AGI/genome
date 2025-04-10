import os
import json
from PIL import Image
from engine.registry import registry
from engine.utils import ProgramInterpreter
from engine.datasets.base_dataset import BaseDataset
from engine.step_interpreters import *


@registry.register_dataset('refcoco')
class RefcocoDataset(BaseDataset):
    @classmethod
    def get_samples(cls, args):
        samples = []
        image_path = os.path.join(args.coco_dir, 'train2014')
        from engine.datasets.refer.refer import REFER
        data_root = args.dataset_dir
        refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
        #refer = REFER(data_root, dataset='refcoco',  splitBy='google')
        ref_ids = refer.getRefIds()
        ref_list = refer.loadRefs(ref_ids[:1000])
        for ann in ref_list:
            image_id = ann['image_id']
            image_name = f"COCO_train2014_000000{image_id}.jpg"
            image_file = os.path.join(image_path, image_name)
            answer_ann = refer.loadAnns(ann['ann_id'])[0]
            bbox = answer_ann['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            samples.append({
                    'id': str(ann['ref_id']),
                    'question': ann['sentences'][0]['sent'],  # 'woman in gray shirt facing camera on right'
                    'answer': json.dumps(bbox),  # '[103.93, 299.99, 238.15, 477.40999999999997]'
                    'image_file': image_file
                }
            )
        # json.dump(samples, open('visualization/refcoco.json','w'))
        return samples

    @classmethod
    def get_interpreter(cls, args):
        API.init_models()
        return ProgramInterpreter(dict(
            LOC=LocInterpreter(),
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter(),
            FILTER_SPATIAL=FILTERSPATIALInterpreter(),
            OVERLAPBOX=OVERLAPBOXInterpreter()
        ))

    @classmethod
    def create_init_state(cls, args, ann):
        img_path = ann['image_file']
        image = Image.open(img_path)
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )
        return init_state

    @classmethod
    def process_result(cls, args, pred_answer, prog_state, ann):
        pred_answer = json.dumps(pred_answer)
        return pred_answer

    @classmethod
    def verify_answer(cls, args, pred_answer, ann):
        def bb_intersection_over_union(boxA, boxB):
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
            if interArea == 0:
                return 0
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
            boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)

            # return the intersection over union value
            return iou
        gt_answer = ann['answer']
        box0 = json.loads(pred_answer)
        box1 = json.loads(gt_answer)
        iou = bb_intersection_over_union(box0, box1)
        return iou >= 0.5


class FILTERSPATIALInterpreter():
    step_name = 'FILTERSPATIAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def execute(self,box_list, location="left"):
        box_list_sorted = sorted(box_list, key=lambda x:x[0])
        if "left" in location:
            box = box_list_sorted[0]
        elif location=="middle":
            box_idx = len(box_list_sorted) //2
            box = box_list_sorted[box_idx]
        elif location=="right":
            box_idx = len(box_list_sorted) //2
            box = box_list_sorted[-1]
        return box


class OVERLAPBOXInterpreter():
    step_name = 'OVERLAPBOX'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def calc_size(self,box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def overlap_box(self,box0,box1):
        x0 = max(box0[0], box1[0])
        y0 = max(box0[1], box1[1])
        x1 = min(box0[2], box1[2])
        y1 = min(box0[3], box1[3])
        if x0 >= x1 or y0 >= y1:
            return [0,0,0,0]
        else:
            return [x0,y0,x1,y1]

    def execute(self,box_list0, box_list1):
        box = [0,0,0,0]
        for box0 in box_list0:
            for box1 in box_list1:
                box2 = self.overlap_box(box0,box1)
                if self.calc_size(box2) > self.calc_size(box):
                    box = box2
        return box
