add_startup_object(
  crt1
  SRC
    start.cpp
  COMPILE_OPTIONS
    -ffreestanding # To avoid compiler warnings about calling the main function.
    -fno-builtin
    -nogpulib # Do not include any GPU vendor libraries.
    -nostdinc
    -x cuda # Use the CUDA toolchain to emit the `_start` kernel.
    --offload-device-only
    --offload-arch=${LIBC_GPU_TARGET_ARCHITECTURE}
)
get_fq_target_name(crt1 fq_name)

# Ensure that clang uses the correct linker for this object type.
target_link_libraries(${fq_name} PUBLIC "--target=${LIBC_GPU_TARGET_TRIPLE}")
