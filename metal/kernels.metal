#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = left[index] + right[index];
    }
}
