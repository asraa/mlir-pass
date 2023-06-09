; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -target-abi ilp32f -mattr=+experimental-zfa < %s \
; RUN:     | FileCheck %s
; RUN: llc -mtriple=riscv64 -target-abi lp64f -mattr=+experimental-zfa < %s \
; RUN:     | FileCheck %s

declare float @llvm.minimum.f32(float, float)

define float @fminm_s(float %a, float %b) nounwind {
; CHECK-LABEL: fminm_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fminm.s fa0, fa0, fa1
; CHECK-NEXT:    ret
  %1 = call float @llvm.minimum.f32(float %a, float %b)
  ret float %1
}

declare float @llvm.maximum.f32(float, float)

define float @fmaxm_s(float %a, float %b) nounwind {
; CHECK-LABEL: fmaxm_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmaxm.s fa0, fa0, fa1
; CHECK-NEXT:    ret
  %1 = call float @llvm.maximum.f32(float %a, float %b)
  ret float %1
}


define float @fround_s_1(float %a) nounwind {
; CHECK-LABEL: fround_s_1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fround.s fa0, fa0, rmm
; CHECK-NEXT:    ret
  %call = tail call float @roundf(float %a) nounwind readnone
  ret float %call
}

declare float @roundf(float) nounwind readnone


define float @fround_s_2(float %a) nounwind {
; CHECK-LABEL: fround_s_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fround.s fa0, fa0, rup
; CHECK-NEXT:    ret
  %call = tail call float @floorf(float %a) nounwind readnone
  ret float %call
}

declare float @floorf(float) nounwind readnone


define float @fround_s_3(float %a) nounwind {
; CHECK-LABEL: fround_s_3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fround.s fa0, fa0, rdn
; CHECK-NEXT:    ret
  %call = tail call float @ceilf(float %a) nounwind readnone
  ret float %call
}

declare float @ceilf(float) nounwind readnone


define float @fround_s_4(float %a) nounwind {
; CHECK-LABEL: fround_s_4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fround.s fa0, fa0, rtz
; CHECK-NEXT:    ret
  %call = tail call float @truncf(float %a) nounwind readnone
  ret float %call
}

declare float @truncf(float) nounwind readnone


define float @fround_s_5(float %a) nounwind {
; CHECK-LABEL: fround_s_5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fround.s fa0, fa0
; CHECK-NEXT:    ret
  %call = tail call float @nearbyintf(float %a) nounwind readnone
  ret float %call
}

declare float @nearbyintf(float) nounwind readnone


define float @froundnx_s(float %a) nounwind {
; CHECK-LABEL: froundnx_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    froundnx.s fa0, fa0
; CHECK-NEXT:    ret
  %call = tail call float @rintf(float %a) nounwind readnone
  ret float %call
}

declare float @rintf(float) nounwind readnone

declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)

define i32 @fcmp_olt_q(float %a, float %b) nounwind strictfp {
; CHECK-LABEL: fcmp_olt_q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fltq.s a0, fa0, fa1
; CHECK-NEXT:    ret
  %1 = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict") strictfp
  %2 = zext i1 %1 to i32
  ret i32 %2
}

define i32 @fcmp_ole_q(float %a, float %b) nounwind strictfp {
; CHECK-LABEL: fcmp_ole_q:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fleq.s a0, fa0, fa1
; CHECK-NEXT:    ret
  %1 = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict") strictfp
  %2 = zext i1 %1 to i32
  ret i32 %2
}
