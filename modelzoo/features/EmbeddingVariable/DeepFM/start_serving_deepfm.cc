#include <iostream>
#include<stdlib.h>
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
    \"checkpoint_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DeepFM/result/\", \
    \"savedmodel_dir\": \"/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DeepFM/savedmodels/1658627534/\" \
  } ";



struct input_format392{
	float I1;
  float I2;
  float I3;
  float I4;
  float I5;
  float I6;
  float I7;
  float I8;
  float I9;
  int I10;
  float I11;
  float I12;
  float I13;
  char* C1;
  char* C2;
  char* C3;
  char* C4;
  char* C5;
  char* C6;
  char* C7;
  char* C8;
  char* C9;
  char* C10;
  char* C11;
  char* C12;
  char* C13;
  char* C14;
  char* C15;
  char* C16;
  char* C17;
  char* C18;
  char* C19;
  char* C20;
  char* C21;
  char* C22;
  char* C23;
  char* C24;
  char* C25;
  char* C26;
};

::tensorflow::eas::ArrayProto get_proto(char* char_input,int dim,::tensorflow::eas::ArrayDataType type){
  ::tensorflow::eas::ArrayShape array_shape;
  array_shape.add_dim(1);
  // array_shape.add_dim(dim);
  ::tensorflow::eas::ArrayProto input;
  input.add_string_val(char_input);
  input.set_dtype(type);
  *(input.mutable_array_shape()) = array_shape;

  return input;

}

::tensorflow::eas::ArrayProto get_proto_f(float char_input,int dim,::tensorflow::eas::ArrayDataType type){
  ::tensorflow::eas::ArrayShape array_shape;
  array_shape.add_dim(1);
  // array_shape.add_dim(1);
  ::tensorflow::eas::ArrayProto input;
  input.add_float_val(char_input);
  input.set_dtype(type);
  *(input.mutable_array_shape()) = array_shape;

  return input;

}



int main(int argc, char** argv) {

  char filepath[] = "/home/deeprec/DeepRec/modelzoo/features/EmbeddingVariable/DeepFM/test.csv";

  // // ------------------------------------------initialize serving model-----------------------------------------
  int state;
  void* model = initialize("", model_config, &state);
  if (state == -1) {
    std::cerr << "initialize error\n";
  }
   
  // // ---------------------------------------prepare serving data from file--------------------------------------
  
  FILE *fp = nullptr;
  char *line, *record;
  char buffer2[1024];
  char delim[] = ",";
  char end[] = "\n";
  int j = 0;
  int rows = 0;
  
  
  // get row number
  if ( (fp = fopen(filepath,"at+")) != nullptr) {
      while ((line = fgets(buffer2, sizeof(buffer2), fp)) != nullptr) {
          rows++;
      }
  }

  fclose(fp);
  
 
  // get rows
  char* all_elems[rows*39];
  int cur_pos = 0;

  if ( (fp = fopen(filepath,"at+")) != nullptr) {

      while ((line = fgets(buffer2, sizeof(buffer2), fp)) != NULL) {
          record = strtok(line, delim);

          while (record != NULL) {
              // only 1 label and 39 feature
              if (j >= 40) break;
              // disragard label 
              if (j == 0) {j++; record = strtok(NULL,delim); continue;}

              char* cur_item = (char*) malloc(sizeof(char)*strlen(record));
              strcpy(cur_item,record);
              if (cur_item[strlen(cur_item)-1] == *end) cur_item[strlen(cur_item)-1] = '\0';
              all_elems[cur_pos] = cur_item;

              cur_pos++;
              record = strtok(NULL, delim);
              j++;

          }
          j = 0;
         
      }
  }

  fclose(fp);

   

  // ----------------------------------------------prepare request input----------------------------------------------------
  for(int ii = 0; ii < rows; ii++ ){
        int start_idx = ii * 39;

        struct input_format392 inputs;
        inputs.I1 =  (float)(atof(all_elems[start_idx]));
        inputs.I2 =  (float)(atof(all_elems[start_idx+1]));
        inputs.I3 =  (float)(atof(all_elems[start_idx+2]));
        inputs.I4 =  (float)(atof(all_elems[start_idx+3]));
        inputs.I5 =  (float)(atof(all_elems[start_idx+4]));
        inputs.I6 =  (float)(atof(all_elems[start_idx+5]));
        inputs.I7 =  (float)(atof(all_elems[start_idx+6]));
        inputs.I8 =  (float)(atof(all_elems[start_idx+7]));
        inputs.I9 =  (float)(atof(all_elems[start_idx+8]));
        inputs.I10 = (float)(atof(all_elems[start_idx+9]));
        inputs.I11 = (float)(atof(all_elems[start_idx+10]));
        inputs.I12 = (float)(atof(all_elems[start_idx+11]));
        inputs.I13 = (float)(atof(all_elems[start_idx+12]));
        inputs.C1 =  (char*)all_elems[start_idx+13];
        inputs.C2 =  (char*)all_elems[start_idx+14];
        inputs.C3 =  (char*)all_elems[start_idx+15];
        inputs.C4 =  (char*)all_elems[start_idx+16];
        inputs.C5 =  (char*)all_elems[start_idx+17];
        inputs.C6 =  (char*)all_elems[start_idx+18];
        inputs.C7 =  (char*)all_elems[start_idx+19];
        inputs.C8 =  (char*)all_elems[start_idx+20];
        inputs.C9 =  (char*)all_elems[start_idx+21];
        inputs.C10 = (char*)all_elems[start_idx+22];
        inputs.C11 = (char*)all_elems[start_idx+23];
        inputs.C12 = (char*)all_elems[start_idx+24];
        inputs.C13 = (char*)all_elems[start_idx+25];
        inputs.C14 = (char*)all_elems[start_idx+26];
        inputs.C15 = (char*)all_elems[start_idx+27];
        inputs.C16 = (char*)all_elems[start_idx+28];
        inputs.C17 = (char*)all_elems[start_idx+29];
        inputs.C18 = (char*)all_elems[start_idx+30];
        inputs.C19 = (char*)all_elems[start_idx+31];
        inputs.C20 = (char*)all_elems[start_idx+32];
        inputs.C21 = (char*)all_elems[start_idx+33];
        inputs.C22 = (char*)all_elems[start_idx+34];
        inputs.C23 = (char*)all_elems[start_idx+35];
        inputs.C24 = (char*)all_elems[start_idx+36];
        inputs.C25 = (char*)all_elems[start_idx+37];
        inputs.C26 = (char*)all_elems[start_idx+38];
   
        // get input type
        ::tensorflow::eas::ArrayDataType dtype_f =
            ::tensorflow::eas::ArrayDataType::DT_FLOAT;

        ::tensorflow::eas::ArrayDataType dtype_s =
            ::tensorflow::eas::ArrayDataType::DT_STRING;

        // input setting
        ::tensorflow::eas::ArrayProto I1 = get_proto_f(inputs.I1,1,dtype_f);
        ::tensorflow::eas::ArrayProto I2 = get_proto_f(inputs.I2,1,dtype_f);
        ::tensorflow::eas::ArrayProto I3 = get_proto_f(inputs.I3,1,dtype_f);
        ::tensorflow::eas::ArrayProto I4 = get_proto_f(inputs.I4,1,dtype_f);
        ::tensorflow::eas::ArrayProto I5 = get_proto_f(inputs.I5,1,dtype_f);
        ::tensorflow::eas::ArrayProto I6 = get_proto_f(inputs.I6,1,dtype_f);
        ::tensorflow::eas::ArrayProto I7 = get_proto_f(inputs.I7,1,dtype_f);
        ::tensorflow::eas::ArrayProto I8 = get_proto_f(inputs.I8,1,dtype_f);
        ::tensorflow::eas::ArrayProto I9 = get_proto_f(inputs.I9,1,dtype_f);
        ::tensorflow::eas::ArrayProto I10 = get_proto_f(inputs.I10,1,dtype_f);
        ::tensorflow::eas::ArrayProto I11 = get_proto_f(inputs.I11,1,dtype_f);
        ::tensorflow::eas::ArrayProto I12 = get_proto_f(inputs.I12,1,dtype_f);
        ::tensorflow::eas::ArrayProto I13 = get_proto_f(inputs.I13,1,dtype_f);
        ::tensorflow::eas::ArrayProto C1 = get_proto(inputs.C1,strlen(inputs.C1),dtype_s);
        ::tensorflow::eas::ArrayProto C2 = get_proto(inputs.C2,strlen(inputs.C2),dtype_s);
        ::tensorflow::eas::ArrayProto C3 = get_proto(inputs.C3,strlen(inputs.C3),dtype_s);
        ::tensorflow::eas::ArrayProto C4 = get_proto(inputs.C4,strlen(inputs.C4),dtype_s);
        ::tensorflow::eas::ArrayProto C5 = get_proto(inputs.C5,strlen(inputs.C5),dtype_s);
        ::tensorflow::eas::ArrayProto C6 = get_proto(inputs.C6,strlen(inputs.C6),dtype_s);
        ::tensorflow::eas::ArrayProto C7 = get_proto(inputs.C7,strlen(inputs.C7),dtype_s);
        ::tensorflow::eas::ArrayProto C8 = get_proto(inputs.C8,strlen(inputs.C8),dtype_s);
        ::tensorflow::eas::ArrayProto C9 = get_proto(inputs.C9,strlen(inputs.C9),dtype_s);
        ::tensorflow::eas::ArrayProto C10 = get_proto(inputs.C10,strlen(inputs.C10),dtype_s);
        ::tensorflow::eas::ArrayProto C11 = get_proto(inputs.C11,strlen(inputs.C11),dtype_s);
        ::tensorflow::eas::ArrayProto C12 = get_proto(inputs.C12,strlen(inputs.C12),dtype_s);
        ::tensorflow::eas::ArrayProto C13 = get_proto(inputs.C13,strlen(inputs.C13),dtype_s);
        ::tensorflow::eas::ArrayProto C14 = get_proto(inputs.C14,strlen(inputs.C14),dtype_s);
        ::tensorflow::eas::ArrayProto C15 = get_proto(inputs.C15,strlen(inputs.C15),dtype_s);
        ::tensorflow::eas::ArrayProto C16 = get_proto(inputs.C16,strlen(inputs.C16),dtype_s);
        ::tensorflow::eas::ArrayProto C17 = get_proto(inputs.C17,strlen(inputs.C17),dtype_s);
        ::tensorflow::eas::ArrayProto C18 = get_proto(inputs.C18,strlen(inputs.C18),dtype_s);
        ::tensorflow::eas::ArrayProto C19 = get_proto(inputs.C19,strlen(inputs.C19),dtype_s);
        ::tensorflow::eas::ArrayProto C20 = get_proto(inputs.C20,strlen(inputs.C20),dtype_s);
        ::tensorflow::eas::ArrayProto C21 = get_proto(inputs.C21,strlen(inputs.C21),dtype_s);
        ::tensorflow::eas::ArrayProto C22 = get_proto(inputs.C22,strlen(inputs.C22),dtype_s);
        ::tensorflow::eas::ArrayProto C23 = get_proto(inputs.C23,strlen(inputs.C23),dtype_s);
        ::tensorflow::eas::ArrayProto C24 = get_proto(inputs.C24,strlen(inputs.C24),dtype_s);
        ::tensorflow::eas::ArrayProto C25 = get_proto(inputs.C25,strlen(inputs.C25),dtype_s);
        ::tensorflow::eas::ArrayProto C26 = get_proto(inputs.C26,strlen(inputs.C26),dtype_s);
        

        // PredictRequest
        ::tensorflow::eas::PredictRequest req;
        req.set_signature_name("serving_default");
        req.add_output_filter("Sigmoid:0");
      
        (*req.mutable_inputs())["I1:0"]  = I1;
        (*req.mutable_inputs())["I2:0"]  = I2;
        (*req.mutable_inputs())["I3:0"]  = I3;
        (*req.mutable_inputs())["I4:0"]  = I4;
        (*req.mutable_inputs())["I5:0"]  = I5;
        (*req.mutable_inputs())["I6:0"]  = I6;
        (*req.mutable_inputs())["I7:0"]  = I7;
        (*req.mutable_inputs())["I8:0"]  = I8;
        (*req.mutable_inputs())["I9:0"]  = I9;
        (*req.mutable_inputs())["I10:0"] = I10;
        (*req.mutable_inputs())["I11:0"] = I11;
        (*req.mutable_inputs())["I12:0"] = I12;
        (*req.mutable_inputs())["I13:0"] = I13;
        (*req.mutable_inputs())["C1:0"]  = C1;
        (*req.mutable_inputs())["C2:0"]  = C2;
        (*req.mutable_inputs())["C3:0"]  = C3;
        (*req.mutable_inputs())["C4:0"]  = C4;
        (*req.mutable_inputs())["C5:0"]  = C5;
        (*req.mutable_inputs())["C6:0"]  = C6;
        (*req.mutable_inputs())["C7:0"]  = C7;
        (*req.mutable_inputs())["C8:0"]  = C8;
        (*req.mutable_inputs())["C9:0"]  = C9;
        (*req.mutable_inputs())["C10:0"] = C10;
        (*req.mutable_inputs())["C11:0"] = C11;
        (*req.mutable_inputs())["C12:0"] = C12;
        (*req.mutable_inputs())["C13:0"] = C13;
        (*req.mutable_inputs())["C14:0"] = C14;
        (*req.mutable_inputs())["C15:0"] = C15;
        (*req.mutable_inputs())["C16:0"] = C16;
        (*req.mutable_inputs())["C17:0"] = C17;
        (*req.mutable_inputs())["C18:0"] = C18;
        (*req.mutable_inputs())["C19:0"] = C19;
        (*req.mutable_inputs())["C20:0"] = C20;
        (*req.mutable_inputs())["C21:0"] = C21;
        (*req.mutable_inputs())["C22:0"] = C22;
        (*req.mutable_inputs())["C23:0"] = C23;
        (*req.mutable_inputs())["C24:0"] = C24;
        (*req.mutable_inputs())["C25:0"] = C25;
        (*req.mutable_inputs())["C26:0"] = C26;
        

        size_t size = req.ByteSizeLong(); 
        void *buffer1 = malloc(size);
        req.SerializeToArray(buffer1, size);

        // ----------------------------------------------process and get feedback---------------------------------------------------
        void* output = nullptr;
        int output_size = 0;
        state = process(model, buffer1, size, &output, &output_size);

        // parse response
        std::string output_string((char*)output, output_size);
        ::tensorflow::eas::PredictResponse resp;
        resp.ParseFromString(output_string);
        std::cout << "process returned state: " << state << ", response: " << resp.DebugString();
  }

 //free memory
  for(int i=0; i < rows;i++){
      free(all_elems[i]);
  }

  return 0;
}

