cmake_minimum_required(VERSION 3.15...3.23)

project(CmakeConfigPackageTests LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ---- Test as standalone project the exported config package ----

if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(myproject CONFIG REQUIRED) # for intro, project_options, ...

  if(NOT TARGET myproject_options)
    message(FATAL_ERROR "Required config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies ----

include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)

find_package(Catch2 3 REQUIRED)

# Provide a simple smoke test to make sure that the CLI works and can display a --help message
add_test(NAME cli.has_help COMMAND intro --help)

# Provide a test to verify that the version being reported from the application
# matches the version given to CMake. This will be important once you package
# your program. Real world shows that this is the kind of simple mistake that is easy
# to make, but also easy to test for.
add_test(NAME cli.version_matches COMMAND intro --version)
set_tests_properties(cli.version_matches PROPERTIES PASS_REGULAR_EXPRESSION "${PROJECT_VERSION}")

add_executable(tests tests.cpp)
target_link_libraries(
  tests
  PRIVATE myproject::myproject_warnings
          myproject::myproject_options
          myproject::sample_library
          Catch2::Catch2WithMain
          )

if(WIN32 AND BUILD_SHARED_LIBS)
#  add_custom_command(
#    TARGET tests
#    PRE_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
#    COMMAND_EXPAND_LISTS)
endif()
set_target_properties (tests PROPERTIES FOLDER tests)

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  tests
  TEST_PREFIX
  "unittests."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX
  .xml)

# Add a file containing a set of constexpr tests
add_executable(constexpr_tests constexpr_tests.cpp)
target_link_libraries(
  constexpr_tests
  PRIVATE myproject::myproject_warnings
          myproject::myproject_options
          myproject::sample_library
          Catch2::Catch2WithMain)

catch_discover_tests(
  constexpr_tests
  TEST_PREFIX
  "constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "constexpr."
  OUTPUT_SUFFIX
  .xml)
set_target_properties (constexpr_tests PROPERTIES FOLDER tests)

# Disable the constexpr portion of the test, and build again this allows us to have an executable that we can debug when
# things go wrong with the constexpr testing
add_executable(relaxed_constexpr_tests constexpr_tests.cpp)
target_link_libraries(
  relaxed_constexpr_tests
  PRIVATE myproject::myproject_warnings
          myproject::myproject_options
          myproject::sample_library
          Catch2::Catch2WithMain)
target_compile_definitions(relaxed_constexpr_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

catch_discover_tests(
  relaxed_constexpr_tests
  TEST_PREFIX
  "relaxed_constexpr."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "relaxed_constexpr."
  OUTPUT_SUFFIX
  .xml)

set_target_properties (relaxed_constexpr_tests PROPERTIES FOLDER tests)

 #
add_executable(tokenizer_tests tokenizer_tests.cpp)
target_link_libraries(
  tokenizer_tests
  PRIVATE myproject::myproject_warnings
          myproject::myproject_options
          tokenizer
          common
          ggml
          spdlog
          fmt
          Catch2::Catch2WithMain)
target_compile_definitions(tokenizer_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

catch_discover_tests(
  tokenizer_tests
  TEST_PREFIX
  "tokenizer."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "tokenizer."
  OUTPUT_SUFFIX
  .xml)
set_target_properties (tokenizer_tests PROPERTIES FOLDER tests)

target_include_directories(tokenizer_tests ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                                                  "${CMAKE_BINARY_DIR}/configured_files/include"
                                                                  )
#
add_executable(inference_tests inference_tests.cpp)
target_link_libraries(
  inference_tests
  PRIVATE myproject::myproject_warnings
          myproject::myproject_options
          tokenizer
          inference
          common
          ggml
          spdlog
          fmt
          Catch2::Catch2WithMain)
target_compile_definitions(inference_tests PRIVATE -DCATCH_CONFIG_RUNTIME_STATIC_REQUIRE)

catch_discover_tests(
  inference_tests
  TEST_PREFIX
  "inference."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "inference."
  OUTPUT_SUFFIX
  .xml)
set_target_properties (inference_tests PROPERTIES FOLDER tests)

target_include_directories(inference_tests ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                                                  "${CMAKE_BINARY_DIR}/configured_files/include"
                                                                  )
