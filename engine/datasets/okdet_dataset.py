import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from PIL import Image
from engine.registry import registry
from engine.utils import ProgramInterpreter
from engine.datasets.base_dataset import BaseDataset
from engine.step_interpreters import *
from scipy.special import softmax


@registry.register_dataset('okdet')
class OKDETDataset():
    @classmethod
    def get_samples(cls, args):
        samples = []
        dataset_dir = args.dataset_dir
        ann_path = os.path.join(dataset_dir, args.ann_path)
        ann_list = json.load(open(ann_path))
        image_path = os.path.join(dataset_dir, args.image_path)
        for index, ann in enumerate(ann_list):
            image_id = ann["image"]  # 'n161313.jpg'
            image_file = os.path.join(image_path, image_id)
            output_file = os.path.join(args.output_dir, f"output_{index}.jpg")
            samples.append({
                    'id': str(index),
                    'question': ann['question'],
                    'answer': ann['answer'],
                    'image_file': image_file,
                    "output_file": output_file
                }
            )
        return samples

    @classmethod
    def get_interpreter(cls, args):
        API.init_models()
        return ProgramInterpreter(dict(
            FACEDET=FaceDetInterpreter(),
            LIST=ListInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            RESULT=ResultInterpreter(),
            TAG=TagInterpreter(),
            LOC=Loc2Interpreter(thresh=0.05,nms_thresh=0.3),
            SORT_SPATIAL_OBJ=SORT_SPATIAL_OBJ(),
            META_COMPARE=META_COMPARE(),
            REDUCE_MASK=REDUCE_MASK(),
        ))

    @classmethod
    def create_init_state(cls, args, ann):
        img_path = ann['image_file']
        image = Image.open(img_path)
        # image.thumbnail((640,640),Image.ANTIALIAS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )
        return init_state
    

    is_init = False

    @classmethod
    def init(cls):
        #@title Sets up the BERT tokenizer using tf-text + helper functions for BEM calculation
        # TODO : fix path
        VOCAB_PATH = 'test/vocab.txt'

        vocab_table = tf.lookup.StaticVocabularyTable(
                tf.lookup.TextFileInitializer(
                    filename=VOCAB_PATH,
                    key_dtype=tf.string,
                    key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                    value_dtype=tf.int64,
                    value_index=tf.lookup.TextFileIndex.LINE_NUMBER
                ),
                num_oov_buckets=1)
        cls.cls_id, cls.sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
        cls.tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_table,
                                    token_out_type=tf.int64,
                                    preserve_unused_token=True,
                                    lower_case=True)
        # TODO : fix path
        cls.bem = hub.load('test/bem')

    @classmethod
    def bertify_example(cls, example):
        question = cls.tokenizer.tokenize(example['question']).merge_dims(1, 2)
        reference = cls.tokenizer.tokenize(example['gt_answer']).merge_dims(1, 2)
        candidate = cls.tokenizer.tokenize(example['pred_answer']).merge_dims(1, 2)

        input_ids, segment_ids = text.combine_segments(
            (candidate, reference, question), cls.cls_id, cls.sep_id)

        return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}

    @classmethod
    def pad(cls, a, length=512):
        return np.append(a, np.zeros(length - a.shape[-1], np.int32))

    @classmethod
    def bertify_examples(cls, examples):
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = cls.bertify_example(example)
            input_ids.append(cls.pad(example_inputs['input_ids']))
            segment_ids.append(cls.pad(example_inputs['segment_ids']))

        return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}

    @classmethod
    def ck_answer(cls, gt_answer, pred_answer, question):
        if not cls.is_init:
            cls.init()
            cls.is_init = True

        inputs = cls.bertify_examples([{
            'question': question,
            'gt_answer': gt_answer,
            'pred_answer': pred_answer
        }])

        # The outputs are raw logits.
        raw_outputs = cls.bem(inputs)

        # They can be transformed into a classification 'probability' like so:
        softmax_score = list(softmax(raw_outputs, axis=1)[:, 1])
        return np.mean(softmax_score)

    @classmethod
    def process_result(cls, args, pred_answer, prog_state, ann):
        # import pdb; pdb.set_trace()
        index = 0
        while f'OBJ{index}' in prog_state:
            index += 1
        index -= 1
        objs = prog_state[f'OBJ{index}']
        ans = ann['answer']

        recall_list = []
        for ans_obj in ans:
            recall = 0
            for obj in objs:
                if bb_intersection_over_union(ans_obj['box'], obj['box']) > 0.5:
                    recall = 1
                    break
            recall_list.append(recall)

        precision_list = []
        for obj in objs:
            precision = 0
            for ans_obj in ans:
                if bb_intersection_over_union(ans_obj['box'], obj['box']) > 0.5:
                    precision = 1
                    break
            precision_list.append(precision)
        loc_res = {'recall': np.mean(recall_list), 'precision': np.mean(precision_list)}

        question = ann['question'].split('Tag')[1]
        recall_list = []
        for ans_obj in ans:
            recall = 0
            for obj in objs:
                if bb_intersection_over_union(ans_obj['box'], obj['box']) > 0.5 \
                    and cls.ck_answer(ans_obj['class'], obj['class'], question) > 0.5:
                    recall = 1
                    break
            recall_list.append(recall)

        precision_list = []
        for obj in objs:
            precision = 0
            for ans_obj in ans:
                if bb_intersection_over_union(ans_obj['box'], obj['box']) > 0.5 \
                    and cls.ck_answer(ans_obj['class'], obj['class'], question) > 0.5:
                    precision = 1
                    break
            precision_list.append(precision)
        tag_res = {'recall': np.mean(recall_list), 'precision': np.mean(precision_list)}

        output_path = ann['output_file']
        pred_answer.save(output_path)
        pred_answer = {'output_path': output_path, 'loc_res': loc_res, 'tag_res': tag_res}
        return pred_answer

    @classmethod
    def verify_answer(cls, args, pred_answer, ann):
        return True

    @classmethod
    def post_process(cls, args):
        pass

class SORT_SPATIAL_OBJ():
    step_name = 'SORT_SPATIAL_OBJ'
    def execute(self,image,object,location,index):
        boxes = [obj['box'] for obj in object]
        # import pdb; pdb.set_trace()
        sort_spatial = self.sub_module_dict['SORT_SPATIAL']
        boxes = [sort_spatial.execute(image,boxes,location,index)]
        # import pdb; pdb.set_trace()
        res_obj = []
        for obj in object:
            if obj['box'] in boxes:
                res_obj.append(obj)
        return res_obj

class META_COMPARE():
    step_name = 'META_COMPARE'
    def execute(self,function_name,image,obj_list,obj_cmp,name1,name2,attribute):
        out_box_list = []
        verifier = self.sub_module_dict[function_name]
        attr="color" if "COLOR" in function_name else "material"
        question="Does the %s has the same %s as the %s?"%(name1,attr,name2)
        # import pdb; pdb.set_trace()
        for OBJ in obj_list:
            flag = verifier.execute(img=image,boxes1=[OBJ['box']],boxes2=[obj_cmp[0]['box']],obj1=name1,obj2=name2,question=question)
            if flag=='yes' and attribute=="same" or (flag=="no" and attribute=="different"):
                out_box_list.append(OBJ)
        return out_box_list


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

class REDUCE_MASK():
    step_name = 'REDUCE_MASK'
    def execute(self,mask_list1,mask_list2):
        res = []
        boxes = [mask['box'] for mask in mask_list2]
        for mask in mask_list1:
            # import pdb; pdb.set_trace()
            reduce = False
            for box in boxes:
                if bb_intersection_over_union(mask['box'], box) > 0.5:
                    reduce = True
                    break
            if not reduce:
                res.append(mask)
        return res