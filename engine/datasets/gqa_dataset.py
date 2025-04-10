import os
import json
import random
from PIL import Image
from engine.registry import registry
from engine.utils import ProgramInterpreter
from engine.datasets.base_dataset import BaseDataset
from engine.step_interpreters import *


@registry.register_dataset('gqa')
class GQADataset(BaseDataset):
    @classmethod
    def get_samples(cls, args):
        samples = []
        dataset_dir = args.dataset_dir
        ann_path = os.path.join(dataset_dir, args.ann_path)
        ann_list = json.load(open(ann_path))
        image_path = os.path.join(dataset_dir, args.image_path)
        if args.random:
            # shuffle
            random.seed(1926)
            random.shuffle(ann_list)
        for ann in ann_list:
            if not args.aimos:
                image_id = ann["imageId"]
                image_file = os.path.join(image_path, image_id + '.jpg')
            else:
                image_id = ann["image"]  # 'n161313.jpg'
                image_file = os.path.join(image_path, image_id)
            samples.append({
                    'id': str(ann['question_id']),
                    'question': ann['question'],
                    'answer': ann['answer'],
                    'image_file': image_file
                }
            )
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
            VERIFY_COLOR=VERIFY_COLOR(),
            COMPARE_COLOR=COMPARE_COLOR(),
            CHOOSE_ATTRIBUTE=CHOOSE_ATTRIBUTE(),
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


class VERIFY_COLOR():
    step_name = 'VERIFY_COLOR'

    def predict(self,img,color):
        colors = ['red','blue','green','yellow','orange','purple','pink','black','white','gray','brown','gold']
        result = API.xvlm(img,color,colors)
        return result

    def execute(self,img,color):
        result = self.predict(img,color)
        if result > 0.6:
            flag = False
        else:
            flag = True
        return flag




class COMPARE_COLOR():
    step_name = 'COMPARE_COLOR'

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def predict(self,img,boxes1,boxes2,obj1,obj2):
        if len(boxes1) > 0:
            box1 = boxes1[0]
            box1 = self.expand_box(box1,img.size)
            out_img1 = img.crop(box1)
        else:
            out_img1 = img

        if len(boxes2) > 0:
            box2 = boxes2[0]
            box2 = self.expand_box(box2,img.size)
            out_img2 = img.crop(box2)
        else:
            out_img2 = img

        color1 = API.vqa(out_img1, f'What color is the {obj1}?')
        color2 = API.vqa(out_img2, f'What color is the {obj2}?')

        prompt = f'Can the {color1} be regarded as the same color as {color2}? You should just reply yes or no without any other words.' 
        temp = API.gpt3(prompt, 'gpt3_general')

        return 'yes' in temp.lower()

    def execute(self,img,boxes1,boxes2,obj1,obj2):
        result = self.predict(img,boxes1,boxes2,obj1,obj2)
        return result


class CHOOSE_ATTRIBUTE():
    step_name = 'CHOOSE_ATTRIBUTE'

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def predict(self,img,boxes,obj,attr1,attr2):
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            out_img = img
        prompt1 = f'Tell me the attributes when the {obj} is {attr1} in one sentence.'
        prompt2 = f'Tell me the attributes when the {obj} is {attr2} in one sentence.'
        obj_desc1 = API.gpt3(prompt1, 'gpt3_general')
        obj_desc2 = API.gpt3(prompt2, 'gpt3_general')
        result1 = API.clip(out_img,obj_desc1)
        result2 = API.clip(out_img,obj_desc2)
        if result1 < result2:
            result = attr1
        else:
            result = attr2
        return result

    def execute(self,img,boxes,obj,attr1,attr2):
        result = self.predict(img,boxes,obj,attr1,attr2)
        return result