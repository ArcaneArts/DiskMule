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

kernel void rms_norm(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    constant uint &weighted [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    float sum = 0.0f;
    for (uint index = lane; index < count; index += lanes) {
        sum = fma(input[index], input[index], sum);
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(scratch[0] / float(count) + epsilon);
    for (uint index = lane; index < count; index += lanes) {
        const float scale = weighted == 0 ? 1.0f : weight[index];
        output[index] = input[index] * scale * inverse_rms;
    }
}

kernel void rope_neox(
    device float *values [[buffer(0)]],
    constant uint &head_dimension [[buffer(1)]],
    constant uint &rotated_pairs [[buffer(2)]],
    constant uint &position [[buffer(3)]],
    constant float &theta [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint pair = index % rotated_pairs;
    const uint head = index / rotated_pairs;
    const uint half_dimension = head_dimension / 2;
    const uint first_index = head * head_dimension + pair;
    const uint second_index = first_index + half_dimension;
    const float exponent = -float(2 * pair) / float(head_dimension);
    const float angle = float(position) * pow(theta, exponent);
    const float sine = sin(angle);
    const float cosine = cos(angle);
    const float first = values[first_index];
    const float second = values[second_index];
    values[first_index] = first * cosine - second * sine;
    values[second_index] = first * sine + second * cosine;
}

kernel void gelu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        const float value = gate[index];
        const float cubic = value * value * value;
        const float gelu = 0.5f * value
            * (1.0f + tanh(0.7978846f * (value + 0.044715f * cubic)));
        output[index] = gelu * up[index];
    }
}

kernel void stable_softmax(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    float maximum = -INFINITY;
    for (uint index = lane; index < count; index += lanes) {
        maximum = max(maximum, input[index]);
    }
    scratch[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum = scratch[0];
    float sum = 0.0f;
    for (uint index = lane; index < count; index += lanes) {
        const float exponential = exp(input[index] - maximum);
        output[index] = exponential;
        sum += exponential;
    }
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_sum = 1.0f / scratch[0];
    for (uint index = lane; index < count; index += lanes) {
        output[index] *= inverse_sum;
    }
}

kernel void logit_softcap(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant float &cap [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = cap * tanh(input[index] / cap);
    }
}

kernel void deterministic_argmax(
    device const float *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best_value = -INFINITY;
    uint best_index = UINT_MAX;
    for (uint index = lane; index < count; index += lanes) {
        const float candidate = input[index];
        if (candidate > best_value || (candidate == best_value && index < best_index)) {
            best_value = candidate;
            best_index = index;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            const float candidate = values[lane + stride];
            const uint candidate_index = indices[lane + stride];
            if (candidate > values[lane]
                || (candidate == values[lane] && candidate_index < indices[lane])) {
                values[lane] = candidate;
                indices[lane] = candidate_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        output[0] = indices[0];
    }
}
