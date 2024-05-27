
set(TARGET common)

file(
    GLOB_RECURSE _source_list
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.inl"
)

list(FILTER _source_list EXCLUDE REGEX "CMakeFiles")
list(FILTER _source_list EXCLUDE REGEX ".*\.cmake")

add_library(${TARGET} STATIC ${_source_list})

foreach(_source IN ITEMS ${_source_list})
    get_filename_component(_source_path "${_source}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${_source_path_msvc}" FILES "${_source}")
endforeach()
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties (${TARGET} PROPERTIES FOLDER libraries)

target_include_directories(${TARGET} ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_link_libraries(
  ${TARGET}
  PRIVATE myproject::myproject_options
          myproject::myproject_warnings)

target_link_system_libraries(
  ${TARGET}
  PRIVATE
          CLI11::CLI11
          fmt::fmt
          spdlog::spdlog
          lefticus::tools
          ggml
          )

target_include_directories(${TARGET} PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include")

