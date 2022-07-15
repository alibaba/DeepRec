#include <iostream>
#include "serving/processor/serving/processor.h"
#include "serving/processor/serving/predict.pb.h"

static const char* model_config = "{ \
    \"omp_num_threads\": 4, \
    \"kmp_blocktime\": 0, \
    \"feature_store_type\": \"memory\", \
    \"serialize_protocol\": \"protobuf\", \
    \"inter_op_parallelism_threads\": 10, \
    \"intra_op_parallelism_threads\": 10, \
    \"init_timeout_minutes\": 1, \
    \"signature_name\": \"serving_default\", \
    \"read_thread_num\": 3, \
    \"update_thread_num\": 2, \
    \"model_store_type\": \"local\", \
    \"checkpoint_dir\": \"/root/deeprec/DeepRec/modelzoo/BST/result/\", \
    \"savedmodel_dir\": \"/root/deeprec/DeepRec/modelzoo/BST/savedmodels/1657183908.2336085/\" \
  } ";

INPUT_FEATURES = [
    'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand',
    'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level',
    'tag_category_list', 'tag_brand_list', 'price'
]

struct input_format{
	string pid;
  string adgroup_id;
  string cate_id;
  string campaign_id;
  string customer;
  string brand;
  string user_id;
  string cms_segid;
  string cms_group_id;
  string final_gender_code;
  string age_level;
  string pvalue_level;
  string shopping_level;
  string occupation;
  string new_user_class_level;
  string tag_category_list;
  string tag_brand_list;
  string price;
	
};

::tensorflow::eas::ArrayProto get_proto(char* char_input,int dim,::tensorflow::eas::ArrayDataType type){
  ::tensorflow::eas::ArrayShape array_shape;
  array_shape.add_dim(1);
  array_shape.add_dim(dim);
  // input array
  ::tensorflow::eas::ArrayProto input;
  input.add_string_val(char_input);
  input.set_dtype(type);
  *(input.mutable_array_shape()) = array_shape;

  return input;

}

int main(int argc, char** argv) {
  int state;
  void* model = initialize("", model_config, &state);
  if (state == -1) {
    std::cerr << "initialize error\n";
  }
   
  // input format
  input_format inputs = {"430548_1007","669310","1665","360359","167792","247789","841908","81","10","1","4","2","3","0","3","8153|8153|8153|8154|8154|8154|1673|1673|1673|6115|6115|6115|1665|1665|1665|1665|1665|1665|8188|8188|8188|8188|8188|8188|8188|8188|8188|1665|1665|1665|8188|8188|8188|8188|8188|8188|8188|8188|8188|10747|10747|10747|10747|10747|10747|10747|10747|10747|10747|10747","197848|197848|197848|237004|237004|237004|330898|330898|330898|337445|337445|337445|258262|258262|258262|247789|247789|247789|339517|339517|339517|339517|339517|339517|339517|339517|339517|278878|278878|278878|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517|339517","6"}

  // input type: float
  ::tensorflow::eas::ArrayDataType dtype =
      ::tensorflow::eas::ArrayDataType::DT_STRING;


// ------------------------------------------------------------------------input setting------------------------------------------------------------------------------
  
  ::tensorflow::eas::ArrayProto input0 = get_proto(inputs.pid,strlen(inputs.pid),dtype);
  ::tensorflow::eas::ArrayProto input1 = get_proto(inputs.adgroup_id,strlen(inputs.adgroup_id),dtype);
  ::tensorflow::eas::ArrayProto input2 = get_proto(inputs.cate_id,strlen(inputs.cate_id),dtype);
  ::tensorflow::eas::ArrayProto input3 = get_proto(inputs.campaign_id,strlen(inputs.campaign_id),dtype);
  ::tensorflow::eas::ArrayProto input4 = get_proto(inputs.customer,strlen(inputs.customer),dtype);
  ::tensorflow::eas::ArrayProto input5 = get_proto(inputs.brand,strlen(inputs.brand),dtype);
  ::tensorflow::eas::ArrayProto input6 = get_proto(inputs.user_id,strlen(inputs.user_id),dtype);
  ::tensorflow::eas::ArrayProto input7 = get_proto(inputs.cms_segid,strlen(inputs.cms_segid),dtype);
  ::tensorflow::eas::ArrayProto input8 = get_proto(inputs.cms_group_id,strlen(inputs.cms_group_id),dtype);
  ::tensorflow::eas::ArrayProto input9 = get_proto(inputs.final_gender_code,strlen(inputs.final_gender_code),dtype);
  ::tensorflow::eas::ArrayProto input10 = get_proto(inputs.age_level,strlen(inputs.age_level),dtype);
  ::tensorflow::eas::ArrayProto input11 = get_proto(inputs.pvalue_level,strlen(inputs.pvalue_level),dtype);
  ::tensorflow::eas::ArrayProto input12 = get_proto(inputs.shopping_level,strlen(inputs.shopping_level),dtype);
  ::tensorflow::eas::ArrayProto input13 = get_proto(inputs.occupation,strlen(inputs.occupation),dtype);
  ::tensorflow::eas::ArrayProto input14 = get_proto(inputs.new_user_class_level,strlen(inputs.new_user_class_level),dtype);
  ::tensorflow::eas::ArrayProto input15 = get_proto(inputs.tag_category_list,strlen(inputs.tag_category_list),dtype);
  ::tensorflow::eas::ArrayProto input16 = get_proto(inputs.tag_brand_list,strlen(inputs.tag_brand_list),dtype);
  ::tensorflow::eas::ArrayProto input17 = get_proto(inputs.price,strlen(inputs.price),dtype);
  

 
  // PredictRequest
  ::tensorflow::eas::PredictRequest req;
  req.set_signature_name("serving_default");
  req.add_output_filter("output:0");
 
  (*req.mutable_inputs())["pid:0"] = input0;
  (*req.mutable_inputs())["adgroup_id:0"] = input1;
  (*req.mutable_inputs())["cate_id:0"] = input2;
  (*req.mutable_inputs())["campaign_id:0"] = input3;
  (*req.mutable_inputs())["customer:0"] = input4;
  (*req.mutable_inputs())["brand:0"] = input5;
  (*req.mutable_inputs())["user_id:0"] = input6;
  (*req.mutable_inputs())["cms_segid:0"] = input7;
  (*req.mutable_inputs())["cms_group_id:0"] = input8;
  (*req.mutable_inputs())["final_gender_code:0"] = input9;
  (*req.mutable_inputs())["age_level:0"] = input10;
  (*req.mutable_inputs())["pvalue_level:0"] = input11;
  (*req.mutable_inputs())["shopping_level:0"] = input12;
  (*req.mutable_inputs())["occupation:0"] = input13;
  (*req.mutable_inputs())["new_user_class_level:0"] = input14;
  (*req.mutable_inputs())["tag_category_list:0"] = input15;
  (*req.mutable_inputs())["tag_brand_list:0"] = input16;
  (*req.mutable_inputs())["price:0"] = input17;

  size_t size = req.ByteSizeLong(); 
  void *buffer = malloc(size);
  req.SerializeToArray(buffer, size);

  // do process
  void* output = nullptr;
  int output_size = 0;
  state = process(model, buffer, size, &output, &output_size);

  // parse response
  std::string output_string((char*)output, output_size);
  ::tensorflow::eas::PredictResponse resp;
  resp.ParseFromString(output_string);
  std::cout << "process returned state: " << state << ", response: " << resp.DebugString();

  return 0;
}

