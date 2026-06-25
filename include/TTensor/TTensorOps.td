//===- TTensorOps.td - TTensor dialect and op definitions --*- tablegen -*-===//
//
// A small, custom front-end dialect with a single high-level op,
// ttensor.relu_add, that desugars into the standard tensor/linalg pipeline.
// It exists to demonstrate defining a dialect with MLIR's ODS (TableGen)
// system, not to add a new abstraction the rest of the pipeline depends on:
// -convert-ttensor-to-linalg expands it away before any other pass runs.
//
//===----------------------------------------------------------------------===//

#ifndef TTENSOR_OPS
#define TTENSOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TTensor_Dialect : Dialect {
  let name = "ttensor";
  let cppNamespace = "::mlir::ttensor";
  let summary = "A small custom front-end dialect for fused tensor ops.";
  let description = [{
    Defines a single high-level op, `ttensor.relu_add`, as a teaching example
    of MLIR's dialect/ODS system. `-convert-ttensor-to-linalg` lowers it into
    the same `tensor.empty` / `linalg.add` / `linalg.fill` / `linalg.max`
    sequence used by `examples/fused_relu_add.mlir`, so every later pipeline
    stage only ever sees standard upstream dialects.
  }];
}

class TTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<TTensor_Dialect, mnemonic, traits>;

def TTensor_ReluAddOp : TTensor_Op<"relu_add", [Pure, SameOperandsAndResultType]> {
  let summary = "Fused relu(lhs + rhs)";
  let description = [{
    Computes `max(lhs + rhs, 0)` element-wise on two equally-shaped tensors
    as a single op, instead of spelling out the add/fill/max sequence by
    hand. `lhs`, `rhs`, and the result must all have the same tensor type.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // TTENSOR_OPS
