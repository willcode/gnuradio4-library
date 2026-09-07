add_library(gnuradio-algorithm INTERFACE)
add_library(gnuradio4::gnuradio-algorithm ALIAS gnuradio-algorithm)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)
# fourier/fft.hpp includes the vendored PocketFFT at the path it is installed at, below gnuradio-4.0/ so it cannot
# collide with a pocketfft the system happens to carry. That path resolves from the include root alone, so a consumer
# that finds this project through pkg-config needs no include directory beyond -I${includedir}. The source tree mirrors
# the installed layout under third_party/include/, which is carried as a system include because the vendored sources do
# not meet this tree's warning battery.
target_include_directories(gnuradio-algorithm SYSTEM
                           INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/include/>)
# GCC cannot prove PocketFFT's plan pointers non-null once its codelets are inlined. The warning is raised by an
# optimizer pass, so a system include does not suppress it; fft.hpp carries the same suppression as a pragma, for
# consumers that do not take their flags from this target. exprtk carries it as a private option.
target_compile_options(gnuradio-algorithm INTERFACE $<$<CXX_COMPILER_ID:GNU>:-Wno-null-dereference>)
install(TARGETS gnuradio-algorithm EXPORT ${GR4_EXPORT_NAME})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/algorithm"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/third_party/include/gnuradio-4.0/third_party/pocketfft"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0/third_party")

# gnuradio-core carries MemoryAllocators/Message/HistoryBuffer/SVD/Tensor/DataSet, which the public headers of this
# target include directly
target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio4::gnuradio-options
            gnuradio4::gnuradio-core
            gnuradio4::gnuradio-meta
            gnuradio4::vir
            gnuradio4::magic_enum)

if(GR_HTTP_ENABLED AND NOT EMSCRIPTEN)
  target_link_libraries(gnuradio-algorithm INTERFACE cpr::cpr)
endif()
target_compile_definitions(gnuradio-algorithm INTERFACE GR_HTTP_ENABLED=$<BOOL:${GR_HTTP_ENABLED}>)

if(ENABLE_EXAMPLES)
  add_subdirectory(benchmarks)
  add_subdirectory(src)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
