#include "Searchers/Recall.h"

std::vector<float *> *Recall::getResult(const std::string &fn, int queryNo, int k) {
  assert(Const::k == k);
  FILE *f = fopen(fn.c_str(), "rb");
  size_t max_k = Const::MAX_TOPK;
  long off = (long)queryNo * (max_k * sizeof(uint32_t));
  fseek(f, off, SEEK_SET);

  std::vector<uint32_t> res_idxs;
  // 先读答案的下标
  //   fread(&res_num, sizeof(int), 1, f);
  //   assert(res_num == k);
  res_idxs.resize(k);
  fread(res_idxs.data(), sizeof(uint32_t), k, f);
  fclose(f);

  // 读对应的 ts
  FILE *data_f = fopen(Const::datafn.c_str(), "r");
  auto *res = new std::vector<float *>(k);
  //   printf("query idx: %d\n", queryNo);
  for (int i = 0; i < k; ++i) {
    // printf("answer idx: %d, ", res_idxs[i]);
    size_t off = (size_t)Const::tsLengthBytes * res_idxs[i];
    fseek(data_f, off, SEEK_SET);
    auto *ts = new float[Const::tsLength];
    fread(ts, sizeof(float), Const::tsLength, data_f);
    (*res)[i] = ts;
  }
  //   printf("\n");
  fclose(data_f);

  return res;
}
