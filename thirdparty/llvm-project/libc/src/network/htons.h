//===-- Implementation header of htons --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NETWORK_HTONS_H
#define LLVM_LIBC_SRC_NETWORK_HTONS_H

#include <stdint.h>

namespace __llvm_libc {

uint16_t htons(uint16_t hostlong);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_NETWORK_HTONS_H
