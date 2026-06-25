//===- TTensorOps.h - TTensor dialect C++ declarations ---------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef TTENSOR_TTENSOROPS_H
#define TTENSOR_TTENSOROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TTensor/TTensorOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TTensor/TTensorOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "TTensor/TTensorOps.h.inc"

#endif // TTENSOR_TTENSOROPS_H
