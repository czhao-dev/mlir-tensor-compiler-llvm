find_program(LIT_EXECUTABLE lit)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY)

if(LIT_EXECUTABLE)
  add_custom_target(check
    COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tensor-pipeline-opt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running tensor-pipeline lit test suite")
else()
  message(STATUS "lit not found; the 'check' target will be unavailable. Install with: pip install lit")
endif()
