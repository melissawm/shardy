/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_OPENXLA_SHARDY_SRC_SHARDY_COMMON_SAVE_MODULE_OP_H_
#define THIRD_PARTY_OPENXLA_SHARDY_SRC_SHARDY_COMMON_SAVE_MODULE_OP_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace sdy {

void saveModuleOp(ModuleOp moduleOp, StringRef dumpDirectory,
                  StringRef fileName);

}  // namespace sdy
}  // namespace mlir

#endif  // THIRD_PARTY_OPENXLA_SHARDY_SRC_SHARDY_COMMON_SAVE_MODULE_OP_H_
