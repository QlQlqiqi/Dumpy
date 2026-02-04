#ifndef MY_TIMER_H
#define MY_TIMER_H

#include <chrono>
#include <cstddef>
#include <utility>
#include <unordered_map>

// 主要是计算各个部分的时间
class MyTimer {
public:
  static auto Now() { return std::chrono::high_resolution_clock::now(); }

  template <typename T>
  static auto
  Duration(const std::chrono::high_resolution_clock::time_point &start,
           const std::chrono::high_resolution_clock::time_point &end) {
    return std::chrono::duration_cast<T>(end - start);
  }

  // search 就是 knn search 时间，包含 approx search 和 exact search
  // 单位 us
  static size_t search_timecount_us_;
  static size_t exact_search_timecount_us_;
  // growIndexFuzzy 耗时
  static size_t node_split_us;
  // node split 时，选择出要分裂的 bit/seg 耗费的时间
  static size_t choose_seg_timecount_us_;
};

// 记录一些次数
class MyCnt {
public:
  // exact search 的 ts 数量
  static size_t exact_search_item_num;
  // build index 时尝试的 plan 数量
  static size_t try_plan_num_;
  // search 所遍历的 internal node 数量
  static size_t exact_search_internal_node_num;
  // search 所遍历的 leaf node 数量
  static size_t exact_search_leaf_node_num;
  // 每个节点划分方案中片段数量
  static std::unordered_map<size_t, size_t> plan_seg_nums_;
};

#endif
