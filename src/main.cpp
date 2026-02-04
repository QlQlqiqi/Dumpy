#include "../include/DataStructures/DumpyNode.h"
#include "../include/DataStructures/GraphConstruction.h"
#include "../include/Searchers/DumpySearcher.h"
#include "../include/Utils/FileUtil.h"
#include "../include/Utils/MathUtil.h"
#include "../include/Utils/TimeSeriesUtil.h"
#include "MyTimer.h"
#include "Searchers/Recall.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>
using namespace std;

vector<vector<int>> *loadGraphSkeleton() {
  int vd = 0;
  for (int i = 1; i <= Const::bitsReserve; ++i)
    vd += MathUtil::nChooseK(Const::segmentNum, i);
  auto nnList = new vector<vector<int>>(Const::vertexNum, vector<int>(vd, -1));

  if (!FileUtil::checkFileExists(Const::graphfn.c_str())) {
    cout << "File not exists!" << Const::graphfn << endl;
    exit(-1);
  }
  FILE *f = fopen(Const::graphfn.c_str(), "rb");

  for (int i = 1; i < Const::vertexNum; ++i)
    fread(&((*nnList)[i][0]), sizeof(int), vd, f);

  return nnList;
}

void constructGraph() { GraphConstruction::buildAndSave2Disk(); }

void buildDumpy() {
  auto g = loadGraphSkeleton();
  DumpyNode *root = DumpyNode::BuildIndex(Const::datafn, Const::saxfn);
  root->save2Disk(Const::idxfn + "root.idx");
}

void approxSearchOneNode() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  auto *g = loadGraphSkeleton();
  float *queries = FileUtil::readQueries();
  auto k = Const::k;

  auto query_num = Const::query_num;

  long duration[query_num];
  double error_ratio[query_num];
  double mAP[query_num];

  for (int i = 0; i < query_num; ++i) {
    auto query = queries + i * Const::tsLength;
    auto start = MyTimer::Now();
    vector<PqItemSeries *> *approxKnn =
        DumpySearcher::approxSearch(root, query, k, g, Const::idxfn);
    auto end = MyTimer::Now();
    MyTimer::search_timecount_us_ +=
        MyTimer::Duration<std::chrono::microseconds>(start, end).count();

    vector<float *> *exactKnn = Recall::getResult(Const::resfn, i, k);

    mAP[i] = TimeSeriesUtil::GetAP(approxKnn, exactKnn);

    vector<PqItemSeries *> exactKnn2;
    for (float *t : *exactKnn) {
      exactKnn2.push_back(new PqItemSeries(t, query));
    }
    error_ratio[i] = MathUtil::errorRatio(*approxKnn, exactKnn2, k);
  }

  printf("total time per search: %zuus\n",
         MyTimer::search_timecount_us_ / query_num);

  double mAP_score = 0;
  for (const auto ap : mAP) {
    mAP_score += ap;
  }
  mAP_score /= query_num;
  printf("mAP is: %.3f\%\n", mAP_score * 100);

  double totalErrorRatio = 0;
  for (double _ : error_ratio)
    totalErrorRatio += _;
  printf("error ratio@%zu is: %.3f\n", k, totalErrorRatio / query_num);

  double total_duration = MyTimer::search_timecount_us_;
  total_duration /= (double)query_num;
  cout << "Average duration is : " << total_duration << "us. "
       << "And QPS = " << 1000000.0 / total_duration << endl;
}

void approxSearchMoreNode() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::approxIncSearch(
        root, queries + i * Const::tsLength, Const::k, Const::idxfn,
        Const::visited_node_num);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void approxSearchOneNodeDTW() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  auto *g = loadGraphSkeleton();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::approxSearchDTW(
        root, queries + i * Const::tsLength, Const::k, g, Const::idxfn);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j) {
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
    }
  }
}

void approxSearchMoreNodeDTW() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::approxIncSearchDTW(
        root, queries + i * Const::tsLength, Const::k, Const::idxfn,
        Const::visited_node_num);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void buildDumpyFuzzy() {
  auto g = loadGraphSkeleton();
  int bound = Const::fuzzy_f * 100;
  Const::fuzzyidxfn +=
      "/" + to_string(bound) + "-" + to_string(Const::delta) + "/";
  DumpyNode *root =
      DumpyNode::BuildIndexFuzzy(Const::datafn, Const::saxfn, Const::paafn, g);
  root->save2Disk(Const::fuzzyidxfn + "root.idx");
}

void approxSearchOneNodeFuzzy() {
  int bound = Const::fuzzy_f * 100;
  Const::fuzzyidxfn +=
      "/" + to_string(bound) + "-" + to_string(Const::delta) + "/";
  DumpyNode *root = DumpyNode::loadFromDisk(
      Const::saxfn, Const::fuzzyidxfn + "root.idx", false);
  auto *g = loadGraphSkeleton();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::approxSearch(
        root, queries + i * Const::tsLength, Const::k, g, Const::fuzzyidxfn);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void approxSearchMoreNodeFuzzy() {
  int bound = Const::fuzzy_f * 100;
  Const::fuzzyidxfn +=
      "/" + to_string(bound) + "-" + to_string(Const::delta) + "/";
  DumpyNode *root = DumpyNode::loadFromDisk(
      Const::saxfn, Const::fuzzyidxfn + "root.idx", false);
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    auto start = chrono::system_clock::now();
    vector<PqItemSeries *> *approxKnn = DumpySearcher::approxIncSearchFuzzy(
        root, queries + i * Const::tsLength, Const::k, Const::fuzzyidxfn,
        Const::visited_node_num);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void exactSearchDumpy() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  auto *g = loadGraphSkeleton();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *exactKnn = DumpySearcher::exactSearch(
        root, queries + i * Const::tsLength, Const::k, g);
    Const::logPrint("Results:");
    for (int j = 0; j < exactKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*exactKnn)[j]->ts) << endl;
  }
}

void exactSearchDumpyDTW() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  auto *g = loadGraphSkeleton();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *exactKnn = DumpySearcher::exactSearchDTW(
        root, queries + i * Const::tsLength, Const::k, g);
    Const::logPrint("Results:");
    for (int j = 0; j < exactKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*exactKnn)[j]->ts) << endl;
  }
}

void ngSearchDumpy() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  root->assignLeafNum();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::ngSearch(
        root, queries + i * Const::tsLength, Const::k, Const::nprobes);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void ngSearchDumpyFuzzy() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  root->assignLeafNum();
  float *queries = FileUtil::readQueries();
  for (int i = 0; i < Const::query_num; ++i) {
    Const::logPrint("Query " + to_string(i) + ":");
    vector<PqItemSeries *> *approxKnn = DumpySearcher::ngSearchFuzzy(
        root, queries + i * Const::tsLength, Const::k, Const::nprobes);
    Const::logPrint("Results:");
    for (int j = 0; j < approxKnn->size(); ++j)
      cout << j + 1 << ": "
           << TimeSeriesUtil::timeSeries2Line((*approxKnn)[j]->ts) << endl;
  }
}

void statIndexDumpy() {
  DumpyNode *root =
      DumpyNode::loadFromDisk(Const::saxfn, Const::idxfn + "root.idx", false);
  root->getIndexStats();
}

void statIndexDumpyFuzzy() {
  int bound = Const::fuzzy_f * 100;
  Const::fuzzyidxfn +=
      "/" + to_string(bound) + "-" + to_string(Const::delta) + "/";
  DumpyNode *root = DumpyNode::loadFromDisk(
      Const::saxfn, Const::fuzzyidxfn + "root.idx", false);
  root->getIndexStats();
}

int main(int argc, char **argv) {
  auto rm_func = [&]() {
    // 删除原本索引
    for (const auto &entry :
         std::filesystem::directory_iterator(Const::idxfn)) {
      std::filesystem::remove_all(entry.path());
    }
    for (const auto &entry :
         std::filesystem::directory_iterator(Const::fuzzyidxfn)) {
      std::filesystem::remove_all(entry.path());
    }
  };

  Const::readConfig();
  if (argc == 2) {
    if (Const::simulate_type == 0) {
      // Dumpy
      Const::index = 1;
      if (strcmp(argv[1], "build") == 0) {
        rm_func();
        Const::ops = 0;
      } else if (strcmp(argv[1], "search") == 0) {
        Const::ops = 1;
      }
    } else if (Const::simulate_type == 3) {
      // Dumpy-Fuzzy
      Const::index = 2;
      if (strcmp(argv[1], "build") == 0) {
        rm_func();
        Const::ops = 0;
      } else if (strcmp(argv[1], "search") == 0) {
        Const::ops = 1;
      }
    }
  }

  SaxUtil::generateSaxFile(Const::datafn, Const::saxfn);
  SaxUtil::generatePaaFile(Const::datafn, Const::paafn);

  switch (Const::index) {
  case 0:
    constructGraph();
    break;
  case 1:
    switch (Const::ops) {
    case 0:
      buildDumpy();
      break;
    case 1:
      approxSearchOneNode();
      break;
    case 2:
      exactSearchDumpy();
      break;
    case 3:
      statIndexDumpy();
      break;
    case 4:
      approxSearchMoreNode();
      break;
    case 5:
      approxSearchOneNodeDTW();
      break;
    case 6:
      approxSearchMoreNodeDTW();
      break;
    case 7:
      ngSearchDumpy();
      break;
    case 8:
      exactSearchDumpyDTW();
      break;
    default:
      break;
    }
    break;
  case 2:
    if (Const::ops == 0) {
      buildDumpyFuzzy();
      break;
    } else if (Const::ops == 1) {
      approxSearchOneNodeFuzzy();
      break;
    } else if (Const::ops == 3) {
      statIndexDumpyFuzzy();
      break;
    } else if (Const::ops == 4) {
      approxSearchMoreNodeFuzzy();
      break;
    } else if (Const::ops == 7) {
      ngSearchDumpyFuzzy();
      break;
    }
    break;
  default:
    break;
  }
}
