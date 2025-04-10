# locate your own data_root, and choose the dataset_splitBy you want to use
from refer import REFER
data_root = "./data"
refer = REFER(data_root, dataset='refcoco',  splitBy='unc')
#refer = REFER(data_root, dataset='refcoco',  splitBy='google')
ref_ids = refer.getRefIds()
ref_list = refer.loadRefs(ref_ids[:10])
import pdb; pdb.set_trace()
