//===- TTensorDialect.cpp - TTensor dialect registration ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "TTensor/TTensorOps.h"

using namespace mlir;
using namespace mlir::ttensor;

#include "TTensor/TTensorOpsDialect.cpp.inc"

void TTensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TTensor/TTensorOps.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "TTensor/TTensorOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TTensor/TTensorOps.cpp.inc"
