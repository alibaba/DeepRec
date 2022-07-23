#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_

#include <pthread.h>
#include <bitset>
#include <atomic>
#include <memory>

#include "tensorflow/core/framework/typed_allocator.h"
#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

namespace tensorflow {

enum class LayoutType {
  LIGHT,
  NORMAL,
  LEVELDB,
  NORMAL_CONTIGUOUS
};

namespace {
constexpr int COLUMN_BITSET_BYTES = 5;
constexpr int COLUMN_BITSET_SIZE = COLUMN_BITSET_BYTES * 8;

struct MetaHeader {
  unsigned char embed_num;
  unsigned char value_type;
  unsigned char header_size;
  unsigned char column_bitset[COLUMN_BITSET_BYTES];

  static const int kEmbeddingNumStartIndex = 0;
  static const int kValueTypeStartIndex =
      kEmbeddingNumStartIndex + sizeof(char);
  static const int kHeaderSizeStartIndex =
      kValueTypeStartIndex + sizeof(char);
  static const int kColumnBitsetIndex =
      kHeaderSizeStartIndex + sizeof(char);

  inline unsigned int GetEmbeddingNum() {
    return (unsigned int) embed_num;
  }

  inline void SetEmbeddingNum(size_t s) {
    embed_num = (unsigned char)s;
  }

  inline std::bitset<COLUMN_BITSET_SIZE> GetColumnBitset() {
    unsigned long meta = ((unsigned long*)this)[0];
    std::bitset<COLUMN_BITSET_SIZE> bs(meta >> (8 * kColumnBitsetIndex));
    return bs;
  }

  inline void SetColumnBitset(const std::bitset<COLUMN_BITSET_SIZE>& bs,
      unsigned int embnum) {
    ((unsigned long*)(this))[0] =
      (bs.to_ulong() << (8 * kColumnBitsetIndex)) |
      (header_size << (8 * kHeaderSizeStartIndex)) |
      (value_type << (8 * kValueTypeStartIndex)) |
      (embnum << (8 * kEmbeddingNumStartIndex));
  }

  inline unsigned int GetHeaderSize() {
    return (unsigned int) header_size;
  }

  inline void SetHeaderSize(size_t size) {
    header_size = (unsigned char)size;
  }

  inline void SetLayoutType(LayoutType vt) {
    value_type = (unsigned char)vt;
  }

  inline LayoutType GetLayoutType() {
    return (LayoutType)value_type;
  }
};

struct LightHeader {
/*__________________________________________________________________________________________
 |           |          |          |               |    embedding     |       slot       |
 | number of | valueptr |  header  | each bit a V* |        V*        |        V*        |
 | embedding | type     |   size   |    1 valid    | actually pointer | actually pointer |...
 |  columns  |          |          |   0 no-valid  |    by alloctor   |    by alloctor   |
 |  (8 bits) | (8 bits) | (8 bits) |   (40 bits)   |     (8 bytes)    |     (8 bytes)    |
 --------------------------------------------------------------------------------------------
*/
  MetaHeader meta;
  LightHeader() {
    memset(this, 0, sizeof(LightHeader));
    meta.SetLayoutType(LayoutType::LIGHT);
    meta.SetHeaderSize(sizeof(LightHeader) / sizeof(int64));
  }
};

struct NormalHeader {
/*_________________________________________________________________________________________________________________________
  |           |          |          |               |             |               |    embedding     |       slot       |
  | number of | valueptr |  header  | each bit a V* | global step | freq counter  |        V*        |        V*        |
  | embedding | type     |   size   |    1 valid    |             |               | actually pointer | actually pointer |...
  |  columns  |          |          |   0 no-valid  |    int64    |     int64     |    by alloctor   |    by alloctor   |
  |  (8 bits) | (8 bits) | (8 bits) |   (40 bits)   |  (8 bytes)  |   (8 bytes)   |     (8 bytes)    |     (8 bytes)    |
  --------------------------------------------------------------------------------------------------------------------------
 */
  MetaHeader meta;
  int64 global_step;
  int64 freq_counter;

  NormalHeader() {
    memset(this, 0, sizeof(NormalHeader));
    meta.SetLayoutType(LayoutType::NORMAL);
    meta.SetHeaderSize(sizeof(NormalHeader) / sizeof(int64));
  }

  inline int64 GetGlobalStep() {
    return global_step;
  }

  inline void SetGlobalStep(int64 gs) {
    global_step = gs;
  }

  inline int64 GetFreqCounter() {
    return freq_counter;
  }

  inline void SetFreqCounter(int64 fc) {
    freq_counter = fc;
  }

  inline void AddFreq() {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + 1);
  }

  inline void AddFreq(int count) {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + count);
  }
};

struct FixedLengthHeader {
/*_________________________________________________________________________________
  |                        |               |                embeddings             |
  | slotflag + global step | freq counter  |                    V                  |
  |                        |               |             actually value            |
  |           int64        |     int64     |               by alloctor             |
  |         (8 bytes)      |   (8 bytes)   |     (4 * slot_num * emb_dim bytes)    |
  ---------------------------------------------------------------------------------
*/
  int64 global_step;
  int64 freq_counter;

  FixedLengthHeader() {
    memset(this, 0, sizeof(FixedLengthHeader));
  }

   inline int64 GetGlobalStep() {
    return global_step & 0x0000ffffffffffff;
  }

  inline void SetGlobalStep(int64 gs) {
    int64 temp = global_step;
    temp &= 0xffff000000000000;
    gs &= 0x0000ffffffffffff;
    temp |= gs;
    global_step = temp;
  }

  inline int64 GetFreqCounter() {
    return freq_counter;
  }

  inline void SetFreqCounter(int64 fc) {
    freq_counter = fc;
  }

  inline void AddFreq() {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + 1);
  }

  inline void AddFreq(int count) {
    __sync_bool_compare_and_swap(&freq_counter,
        freq_counter, freq_counter + count);
  }
};
} // namespace

template <class V>
class ValuePtr {
 public:
  virtual ~ValuePtr() {}

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, int offset) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();

    if (!metadata.test(emb_index)) {
      while(flag_.test_and_set(std::memory_order_acquire));
      metadata = meta->GetColumnBitset();
      if (metadata.test(emb_index)){
        flag_.clear(std::memory_order_release);
        return ((V**)((int64*)ptr_ + (unsigned int)meta->header_size))[emb_index];
      }
      embnum++ ;
      int64 alloc_value_len = value_len;
      V* tensor_val = (V*)allocator->AllocateRaw(0/*alignment unused*/, sizeof(V) * alloc_value_len);
      memcpy(tensor_val, default_v, sizeof(V) * value_len);
      ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index]  = tensor_val;

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      meta->SetColumnBitset(metadata, embnum);
      flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      return ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index];
    }
  }

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, int offset, bool &need_initialize) {

  }

  // simple getter for V* and version
  virtual V* GetValue(int emb_index, int offset) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    auto metadata = meta->GetColumnBitset();
    if (metadata.test(emb_index)) {
      return ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[emb_index];
    } else {
      return nullptr;
    }
  }

  virtual void Free(const V* v) {}

  virtual void Destroy(Allocator* allocator) {
    MetaHeader* meta = (MetaHeader*)ptr_;
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();
    for (int i = 0; i< embnum; i++) {
      if (metadata.test(i)) {
        V* val = ((V**)((int64*)ptr_ + meta->GetHeaderSize()))[i];
        if (val != nullptr) {
          allocator->DeallocateRaw(val);
        }
      }
    }
  }

  virtual void* GetPtr() const {
    return this->ptr_;
  }

  // Global Step
  virtual int64 GetStep() {
    LOG(FATAL) << "Unsupport GlobalStep in subclass of ValuePtrBase";
  }

  virtual void SetStep(int64 gs) {
    LOG(FATAL) << "Unsupport GlobalStep in subclass of ValuePtrBase";
  }

  // Frequency Counter
  virtual int64 GetFreq() {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void SetFreq(int64 freq) {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void AddFreq() {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void AddFreq(int count) {
    LOG(FATAL) << "Unsupport FreqCounter in subclass of ValuePtrBase";
  }

  virtual void SetValue(V val, size_t size){
    LOG(FATAL) << "Unsupport SetValue in subclass of ValuePtrBase";
  }

 protected:
  void* ptr_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

template <class V>
class LightValuePtr : public ValuePtr<V> {
 public:
  LightValuePtr(Allocator* allocator, size_t size) {
    this->ptr_ = (void*) malloc(sizeof(LightHeader) + sizeof(int64) * size);
    memset(this->ptr_ + sizeof(LightHeader), 0, sizeof(int64) * size);
    new ((char*)this->ptr_) LightHeader();
  }

  ~LightValuePtr() {
    free(this->ptr_);
  }
};

template <class V>
class NormalValuePtr : public ValuePtr<V> {
 public:
  NormalValuePtr(Allocator* allocator, size_t size) {
    this->ptr_ = (void*) malloc(sizeof(NormalHeader) + sizeof(int64) * size);
    memset(this->ptr_ + sizeof(NormalHeader), 0, sizeof(int64) * size);
    new ((char*)this->ptr_) NormalHeader();
  }

  ~NormalValuePtr() {
    free(this->ptr_);
  }

  int64 GetStep() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->GetGlobalStep();
  }

  void SetStep(int64 gs) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    ((NormalHeader*)this->ptr_)->SetGlobalStep(gs);
  }

  int64 GetFreq() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->GetFreqCounter();
  }

  void SetFreq(int64 freq) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    ((NormalHeader*)this->ptr_)->SetFreqCounter(freq);
  }

  void AddFreq() {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->AddFreq();
  }

  void AddFreq(int count) {
    MetaHeader* meta = (MetaHeader*)this->ptr_;
    return ((NormalHeader*)this->ptr_)->AddFreq(count);
  }
};

template <class V>
class NormalContiguousValuePtr : public ValuePtr<V>{
  public:
   NormalContiguousValuePtr(Allocator* allocator, size_t size) {
    this->ptr_ = allocator->AllocateRaw(0/*alignemnt unused*/, sizeof(FixedLengthHeader) + sizeof(V) * size);
    memset(this->ptr_ + sizeof(FixedLengthHeader), 0, sizeof(V) * size);
    new ((char*)this->ptr_) FixedLengthHeader();
   }

   ~NormalContiguousValuePtr(){
   }

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, int offset) override {
    int8 meta = *((int8*)((char*)this->ptr_ + 6));
    std::bitset<8> bs(meta);
    if (!bs.test(emb_index)) {
      while(this->flag_.test_and_set(std::memory_order_acquire));
      if (bs.test(emb_index)) {
        return ((V*)this->ptr_ + sizeof(FixedLengthHeader) / sizeof(V) + offset);
      }
      V* tensor_val = ((V*)this->ptr_ + sizeof(FixedLengthHeader) / sizeof(V) + offset);
      memcpy(tensor_val, default_v, sizeof(V) * value_len);
      int8* m = (int8*)((char*)this->ptr_ + 6);
      *m |= (1 <<  emb_index);
      this->flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      return (V*)this->ptr_ + sizeof(FixedLengthHeader) / sizeof(V) + offset;
    }
  }

  virtual V* GetValue(int emb_index, int offset) {
    int8 meta = *((int8*)((char*)this->ptr_ + 6));
    std::bitset<8> bs(meta);
    if (bs.test(emb_index)) {
      return ((V*)this->ptr_ + sizeof(FixedLengthHeader) / sizeof(V) + offset);
    } else {
      return nullptr;
    }
  }

  virtual void Destroy(Allocator* allocator) {
    allocator->DeallocateRaw(this->ptr_);
  }

  int64 GetStep() {
    return ((FixedLengthHeader*)this->ptr_)->GetGlobalStep();
  }

  void SetStep(int64 gs) {
    ((FixedLengthHeader*)this->ptr_)->SetGlobalStep(gs);
  }

  int64 GetFreq() {
    return ((FixedLengthHeader*)this->ptr_)->GetFreqCounter();
  }

  void SetFreq(int64 freq) {
    ((FixedLengthHeader*)this->ptr_)->SetFreqCounter(freq);
  }

  void AddFreq() {
    ((FixedLengthHeader*)this->ptr_)->AddFreq();
  }

  void AddFreq(int count) {
    ((FixedLengthHeader*)this->ptr_)->AddFreq(count);
  }

  void SetValue(V val, size_t size){
    for (int i = 0; i < size; ++i) {
      *((V*)this->ptr_ + sizeof(FixedLengthHeader) / sizeof(V) + i) = val;
    }
  }
};

template <class V>
class NormalGPUValuePtr : public ValuePtr<V> {
 public:
  NormalGPUValuePtr(Allocator* allocator, size_t size) {
    this->ptr_ = (void*) malloc(sizeof(FixedLengthHeader) + sizeof(V *));
    alloc_ = allocator;
    *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) = (V*)allocator->AllocateRaw(0/*alignment unused*/, sizeof(V) * size);
    new ((char*)this->ptr_) FixedLengthHeader();
  }

  ~NormalGPUValuePtr() {
    free(this->ptr_);
  }

#if GOOGLE_CUDA
#if !TF_ENABLE_GPU_EV
  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, int offset) override {
    int8 meta = *((int8*)((char*)this->ptr_ + 6));
    std::bitset<8> bs(meta);
    if (!bs.test(emb_index)) {
      while(this->flag_.test_and_set(std::memory_order_acquire));
      if (bs.test(emb_index))
        return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
      V* tensor_val = *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
      cudaMemcpy(tensor_val, default_v, value_len * sizeof(V), cudaMemcpyDeviceToDevice);
      int8* m = (int8*)((char*)this->ptr_ + 6);
      *m |= (1 <<  emb_index);
      this->flag_.clear(std::memory_order_release);
      return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
    } else {
      return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
    }
  }
#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA

  virtual V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, int offset, bool &need_initialize) override {
    int8 meta = *((int8*)((char*)this->ptr_ + 6));
    std::bitset<8> bs(meta);
    if (!bs.test(emb_index)) {
      while(this->flag_.test_and_set(std::memory_order_acquire));
      if (bs.test(emb_index))
        return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
      V* tensor_val = *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
      need_initialize = 1;
      int8* m = (int8*)((char*)this->ptr_ + 6);
      *m |= (1 <<  emb_index);
      this->flag_.clear(std::memory_order_release);
      return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
    } else {
      return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
    }
  }

  // simple getter for V* and version
  virtual V* GetValue(int emb_index, int offset) {
    int8 meta = *((int8*)((char*)this->ptr_ + 6));
    std::bitset<8> bs(meta);
    if (bs.test(emb_index)) {
      return *(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)) + offset;
    } else {
      return nullptr;
    }
  }

  virtual void Destroy(Allocator* allocator) {
    alloc_->DeallocateRaw(*(V**)((char *)this->ptr_ + sizeof(FixedLengthHeader)));
  }

  int64 GetStep() {
    return ((FixedLengthHeader*)this->ptr_)->GetGlobalStep();
  }

  void SetStep(int64 gs) {
    ((FixedLengthHeader*)this->ptr_)->SetGlobalStep(gs);
  }

  int64 GetFreq() {
    return ((FixedLengthHeader*)this->ptr_)->GetFreqCounter();
  }

  void SetFreq(int64 freq) {
    ((FixedLengthHeader*)this->ptr_)->SetFreqCounter(freq);
  }

  void AddFreq() {
    ((FixedLengthHeader*)this->ptr_)->AddFreq();
  }

  void AddFreq(int64 count) {
    ((FixedLengthHeader*)this->ptr_)->AddFreq(count);
  }
  private:
    Allocator *alloc_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_VALUE_PTR_H_
