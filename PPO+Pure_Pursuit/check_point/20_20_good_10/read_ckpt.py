
from tensorflow.python import pywrap_tensorflow
import os

checkpoint_path = os.path.join("ppo.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

##############################################################################################


# from tensorflow.python.tools import inspect_checkpoint as chkp

# chkp.print_tensors_in_checkpoint_file(file_name="/tmp/model.ckpt", 
#                                       tensor_name, # 如果为None,则默认为ckpt里的所有变量
#                                       all_tensors, # bool 是否打印所有的tensor，这里打印出的是tensor的值，一般不推荐这里设置为False
#                                       all_tensor_names) # bool 是否打印所有的tensor的name
