add_mlir_dialect_library(MLIRTTensor
  TTensorDialect.cpp
  ConvertTTensorToLinalg.cpp

  DEPENDS
  MLIRTTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
)
