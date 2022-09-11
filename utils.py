import numpy as np

def operate_dict(list_dic, op):
  temp = {k:v for k,v in list_dic[0].items()}
  for dic in list_dic[1:]:
    for k,v in dic.items():
      if type(v) is int or type(v) is float or type(v) is np.float64:
        temp[k] = op(temp[k], v)
      elif type(v) is dict:
        temp[k] = operate_dict([temp[k], v], op)
      elif np.isnan(v):
        temp[k] = v
      else:
        print(type(v))
        raise ValueError('Has to be eihter a dictionary or a numerical value')
  return temp

def operate_dict_const(dic, op, const):
  temp = {k:v for k,v in dic.items()}
  for k,v in temp.items():
    if type(v) is int or type(v) is float or type(v) is np.float64:
      temp[k] = op(v, const)
    elif type(v) is dict:
      temp[k] = operate_dict_const(v, op, const)
    elif np.isnan(v):
      temp[k] = v
    else:
      raise ValueError('Has to be eihter a dictionary or a numerical value')
  
  return temp

def filter_metrics_means(scores):
  temp = {}
  for k,v in scores.items():
    temp[k] = {
        'accuracy':v['means']['accuracy'],
        'auroc_score':v['means']['auroc_score'],
        'average_precision':v['means']['average_precision'],
        'f1-score':v['means']['link']['f1-score'],
        'precision':v['means']['link']['precision'],
        'recall':v['means']['link']['recall'],
    }
  return temp

def filter_metrics_sdterr(scores):
  temp = {}
  for k,v in scores.items():
    temp[k] = {
        'accuracy_stderr':v['std_error']['accuracy'],
        'auroc_score_stderr':v['std_error']['auroc_score'],
        'average_precision_stderr':v['std_error']['average_precision'],
        'f1-score_stderr':v['std_error']['link']['f1-score'],
        'precision_stderr':v['std_error']['link']['precision'],
        'recall_stderr':v['std_error']['link']['recall'],
    }
  return temp