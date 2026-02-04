#include "../include/MyTimer.h"

size_t MyTimer::search_timecount_us_ = 0;
size_t MyTimer::exact_search_timecount_us_ = 0;
size_t MyTimer::node_split_us = 0;
size_t MyTimer::choose_seg_timecount_us_ = 0;

size_t MyCnt::exact_search_item_num = 0;
size_t MyCnt::try_plan_num_ = 0;
size_t MyCnt::exact_search_internal_node_num = 0;
size_t MyCnt::exact_search_leaf_node_num = 0;
std::unordered_map<size_t, size_t> MyCnt::plan_seg_nums_;
