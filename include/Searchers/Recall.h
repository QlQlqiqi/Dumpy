#ifndef MULGIFT_RECALL_H
#define MULGIFT_RECALL_H

#include "Const.h"
#include <vector>

class Recall {

public:
  static std::vector<float *> *getResult(const std::string &fn, int queryNo,
                                         int k);
};

#endif
