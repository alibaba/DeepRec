from tensorflow.python.saved_model import loader_impl
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging

source_dir="/home/deeprec/DeepRec/modelzoo/DeepFM/savedmodels/1657790257"

logging.info("before _parse_saved_model.")
saved_model = loader_impl._parse_saved_model(source_dir)
logging.info("_parse_saved_model done.")

path = source_dir + "/saved_model.pb"
# write pbtxt graph 
file_io.write_string_to_file(path+"txt", str(saved_model))
