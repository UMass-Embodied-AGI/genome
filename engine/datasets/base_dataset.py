import os
import json
from engine.registry import registry
from engine.utils import ProgramInterpreter
from engine.step_interpreters import *


@registry.register_dataset('base')
class BaseDataset():
    @classmethod
    def get_samples(cls, args):
        samples = []
        dataset_dir = args.dataset_dir
        ann_path = os.path.join(dataset_dir, args.ann_path)
        ann_list = json.load(open(ann_path))
        for ann in ann_list:
            samples.append({
                    'id': str(ann['question_id']),
                    'question': ann['question'],
                    'answer': ann['answer'],
                }
            )
        return samples

    @classmethod
    def get_interpreter(cls, args):
        API.init_models()
        return ProgramInterpreter(dict(
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        ))

    @classmethod
    def create_init_state(cls, args, ann):
        init_state = {}
        return init_state

    @classmethod
    def process_result(cls, args, pred_answer, prog_state, ann):
        return pred_answer

    @classmethod
    def verify_answer(cls, args, pred_answer, ann):
        gt_answer = ann['answer']
        return str(pred_answer).strip() == str(gt_answer).strip()

    @classmethod
    def post_process(cls, args):
        pass
