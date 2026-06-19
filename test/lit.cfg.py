import os
import lit.formats

config.name = "tensor-pipeline"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tensor_pipeline_obj_root, "test")

config.substitutions.append(("%tensor-pipeline-opt", config.tensor_pipeline_opt))
config.substitutions.append(("FileCheck", config.filecheck))

llvm_bin_dir = os.path.dirname(config.filecheck)
config.environment["PATH"] = os.path.pathsep.join(
    [llvm_bin_dir, config.tensor_pipeline_tools_dir, config.environment["PATH"]]
)
