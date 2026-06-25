get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTTensor
)

add_llvm_executable(tensor-pipeline-opt tensor-pipeline-opt.cpp)
llvm_update_compile_flags(tensor-pipeline-opt)
target_link_libraries(tensor-pipeline-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(tensor-pipeline-opt)
