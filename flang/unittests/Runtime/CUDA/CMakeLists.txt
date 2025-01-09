add_flangrt_unittest(FlangCufRuntimeTests
  Allocatable.cpp
  AllocatorCUF.cpp
  Memory.cpp
)

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  CufRuntime
)
