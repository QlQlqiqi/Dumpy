add_subdirectory(SearchEngine)
add_subdirectory(Utils)
add_subdirectory(IndexConstruction)

add_library(
  dumpy_lib
  OBJECT
  util.cpp
  Const.cpp
  PqItemSeries.cpp
  MyTimer.cpp
)
