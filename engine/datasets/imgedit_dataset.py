import os
import json
from engine.registry import registry
from engine.utils import ProgramInterpreter
from engine.step_interpreters import *


@registry.register_dataset('imgedit')
class IMGEDITDataset():
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
                    'answer': "",
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
            SEG=SegmentInterpreter(),
            SELECT=SelectInterpreter(),
            COLORPOP=ColorpopInterpreter(),
            BGBLUR=BgBlurInterpreter(),
            REPLACE=ReplaceInterpreter(),
            EMOJI=EmojiInterpreter(),
            RESULT=ResultInterpreter(),
            MASK2BOX=MASK2BOX(),
            BOX2MASK=BOX2MASK(),
            LOC=LocInterpreter_glip(),
            META_VERIFY=META_VERIFY(),
            META_COMPARE=META_COMPARE(),
            REDUCE_MASK=REDUCE_MASK(),
        ))

    @classmethod
    def create_init_state(cls, args, ann):
        img_path = ann['image_file']
        image = Image.open(img_path)
        # image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )
        return init_state

    @classmethod
    def process_result(cls, args, pred_answer, prog_state, ann):
        output_path = ann['output_file']
        pred_answer.save(output_path)
        pred_answer = output_path
        return pred_answer

    @classmethod
    def verify_answer(cls, args, pred_answer, ann):
        return True

    @classmethod
    def post_process(cls, args):
        pass
    

class MASK2BOX():
    step_name = 'MASK2BOX'
    def execute(self,MASKLIST):
        BOX_LIST=[mask['box'] for mask in MASKLIST]
        return BOX_LIST
    
class BOX2MASK():
    step_name = 'BOX2MASK'
    def execute(self,MASKLIST,BOX):
        if len(BOX)==0:
            return [MASKLIST[0]]
    
        if not isinstance(BOX[0], list): 
            BOX = [BOX]  
        out_mask_list = [] 
        for box in BOX:
            max_iou = 0
            max_mask = None
            for mask in MASKLIST:
                iou = bb_intersection_over_union(mask["box"], box)
                if iou > max_iou:
                    max_mask = mask
                    max_iou = iou
            out_mask_list.append(max_mask)
        return out_mask_list
        
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

class META_VERIFY():
    step_name = 'META_VERIFY'
    
    def execute(self,function_name,image,obj_list,attribute):
        out_box_list = []
        verifier = self.sub_module_dict[function_name]
        for OBJ in obj_list:
            if "COLOR" in function_name:
                flag = verifier.execute(img=image,boxes=[OBJ['box']],color=attribute,question="Is it %s"%attribute)
            else:
                flag = verifier.execute(img=image,boxes=[OBJ['box']],obj=attribute,question="Is there %s"%attribute)
            if flag:
                out_box_list.append(OBJ)
        import pdb
        pdb.set_trace()
        return out_box_list

class META_COMPARE():
    step_name = 'META_COMPARE'
    def execute(self,function_name,image,obj_list,obj_cmp,name1,name2,attribute):
        out_box_list = []
        verifier = self.sub_module_dict[function_name]
        attr="color" if "COLOR" in function_name else "material"
        question="Does the %s has the same %s as the %s?"%(name1,attr,name2)
        for OBJ in obj_list:
            flag = verifier.execute(img=image,boxes1=[OBJ['box']],boxes2=[obj_cmp[0]['box']],obj1=name1,obj2=name2,question=question)
            if flag=='yes' and attribute=="same" or (flag=="no" and attribute=="different"):
                out_box_list.append(OBJ)
        return out_box_list


class REDUCE_MASK():
    step_name = 'REDUCE_MASK'
    def execute(self,mask_list1,mask_list2):
        res = []
        boxes = [mask['box'] for mask in mask_list2]
        for mask in mask_list1:
            # import pdb; pdb.set_trace()
            if mask['box'] not in boxes:
                res.append(mask)
        return res


class LocInterpreter_glip():
    """
    Input:
        img: an image object
        obj_name: an object string
    Output:
        selected_boxes: a list of bounding boxes
    """

    step_name = 'LOC'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def predict(self,img,obj_name):
        # return API.loc(img,obj_name,self.thresh,self.nms_thresh)
        return API.find(img,obj_name,glip_thresh=0.5)

    def top_box(self,img):
        w,h = img.size
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def execute(self,img,obj_name):
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img,obj_name)
        return bboxes