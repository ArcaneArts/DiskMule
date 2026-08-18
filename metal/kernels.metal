#include <metal_stdlib>
using namespace metal;

inline ushort load_u16_le(device const uchar *bytes) {
    return ushort(bytes[0]) | (ushort(bytes[1]) << 8);
}

inline uint load_u32_le(device const uchar *bytes) {
    return uint(bytes[0])
        | (uint(bytes[1]) << 8)
        | (uint(bytes[2]) << 16)
        | (uint(bytes[3]) << 24);
}

inline float load_f16(device const uchar *bytes) {
    const ushort bits = ushort(bytes[0]) | (ushort(bytes[1]) << 8);
    return float(as_type<half>(bits));
}

inline uint q4_k_scale(device const uchar *scales, uint index) {
    if (index < 4) {
        return uint(scales[index] & 0x3f);
    }
    return uint((scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4));
}

inline uint q4_k_minimum(device const uchar *scales, uint index) {
    if (index < 4) {
        return uint(scales[index + 4] & 0x3f);
    }
    return uint((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

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
        const float argument = 0.7978846f * (value + 0.044715f * cubic);
        const float gelu = 0.5f * value
            * (1.0f + tanh(clamp(argument, -10.0f, 10.0f)));
        output[index] = gelu * up[index];
    }
}

kernel void silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        const float value = gate[index];
        output[index] = (value / (1.0f + exp(-value))) * up[index];
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
    uint best_index = 0xffffffffu;
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

kernel void matvec_f32(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *weights = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const float weight = as_type<float>(load_u32_le(weights + column * 4));
        sum = fma(weight, input[column], sum);
    }
    output[row] = sum;
}

kernel void matvec_f16(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *weights = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const half weight = as_type<half>(load_u16_le(weights + column * 2));
        sum = fma(float(weight), input[column], sum);
    }
    output[row] = sum;
}

kernel void matvec_bf16(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *weights = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const float weight = as_type<float>(uint(load_u16_le(weights + column * 2)) << 16);
        sum = fma(weight, input[column], sum);
    }
    output[row] = sum;
}

inline float decode_f8_e4m3fn(uchar value) {
    const float sign = (value & 0x80) == 0 ? 1.0f : -1.0f;
    const uint exponent = (uint(value) >> 3) & 0x0f;
    const uint fraction = uint(value) & 0x07;
    if (exponent == 0x0f && fraction == 0x07) {
        return NAN;
    }
    const float magnitude = exponent == 0
        ? float(fraction) * 0.001953125f
        : (1.0f + float(fraction) * 0.125f) * exp2(float(int(exponent) - 7));
    return sign * magnitude;
}

kernel void matvec_f8_e4m3fn(
    device const uchar *encoded [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &scale_columns [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *weights = encoded + row * columns;
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const uint scale_index = (row / 128) * scale_columns + column / 128;
        const float weight = decode_f8_e4m3fn(weights[column]) * scales[scale_index];
        sum = fma(weight, input[column], sum);
    }
    output[row] = sum;
}

kernel void matvec_i8_row(
    device const uchar *encoded [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *weights = encoded + row * columns;
    const float scale = scales[row];
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const int quantized = weights[column] < 128
            ? int(weights[column])
            : int(weights[column]) - 256;
        sum = fma(float(quantized) * scale, input[column], sum);
    }
    output[row] = sum;
}

kernel void matvec_i4_grouped(
    device const uchar *encoded [[buffer(0)]],
    device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &scale_columns [[buffer(5)]],
    constant uint &group_size [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    const uint row_bytes = (columns + 1) / 2;
    device const uchar *weights = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint column = 0; column < columns; column++) {
        const uchar packed = weights[column / 2];
        const uint nibble = (column & 1) == 0
            ? uint(packed & 0x0f)
            : uint(packed >> 4);
        const uint scale_index = row * scale_columns + column / group_size;
        const float weight = float(int(nibble) - 8) * scales[scale_index];
        sum = fma(weight, input[column], sum);
    }
    output[row] = sum;
}

kernel void matvec_q5_0(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *row_data = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint block_index = 0; block_index < columns / 32; block_index++) {
        device const uchar *block = row_data + block_index * 22;
        const float delta = load_f16(block);
        const uint high_bits = uint(block[2]) | (uint(block[3]) << 8)
            | (uint(block[4]) << 16) | (uint(block[5]) << 24);
        const uint base = block_index * 32;
        for (uint index = 0; index < 16; index++) {
            const uchar packed = block[6 + index];
            const uint low_high_bit = ((high_bits >> index) << 4) & 0x10;
            const int low = int((uint(packed) & 0x0f) | low_high_bit) - 16;
            sum = fma(delta * float(low), input[base + index], sum);
        }
        for (uint index = 0; index < 16; index++) {
            const uchar packed = block[6 + index];
            const uint high_high_bit = (high_bits >> (index + 12)) & 0x10;
            const int high = int((uint(packed) >> 4) | high_high_bit) - 16;
            sum = fma(delta * float(high), input[base + index + 16], sum);
        }
    }
    output[row] = sum;
}

kernel void matvec_q8_0(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *row_data = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint block_index = 0; block_index < columns / 32; block_index++) {
        device const uchar *block = row_data + block_index * 34;
        const float delta = load_f16(block);
        const uint base = block_index * 32;
        for (uint index = 0; index < 32; index++) {
            const char quant = as_type<char>(block[2 + index]);
            sum = fma(delta * float(quant), input[base + index], sum);
        }
    }
    output[row] = sum;
}

kernel void matvec_q4_k(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *row_data = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint block_index = 0; block_index < columns / 256; block_index++) {
        device const uchar *block = row_data + block_index * 144;
        const float delta = load_f16(block);
        const float minimum = load_f16(block + 2);
        device const uchar *scales = block + 4;
        device const uchar *quants = block + 16;
        const uint block_base = block_index * 256;
        for (uint group_pair = 0; group_pair < 4; group_pair++) {
            const float scale_low = delta * float(q4_k_scale(scales, group_pair * 2));
            const float min_low = minimum * float(q4_k_minimum(scales, group_pair * 2));
            const float scale_high = delta * float(q4_k_scale(scales, group_pair * 2 + 1));
            const float min_high = minimum * float(q4_k_minimum(scales, group_pair * 2 + 1));
            const uint quant_offset = group_pair * 32;
            const uint value_offset = block_base + group_pair * 64;
            for (uint index = 0; index < 32; index++) {
                const uchar packed = quants[quant_offset + index];
                const float low = scale_low * float(packed & 0x0f) - min_low;
                sum = fma(low, input[value_offset + index], sum);
            }
            for (uint index = 0; index < 32; index++) {
                const uchar packed = quants[quant_offset + index];
                const float high = scale_high * float(packed >> 4) - min_high;
                sum = fma(high, input[value_offset + index + 32], sum);
            }
        }
    }
    output[row] = sum;
}

kernel void matvec_q6_k(
    device const uchar *encoded [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &row_bytes [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    device const uchar *row_data = encoded + row * row_bytes;
    float sum = 0.0f;
    for (uint block_index = 0; block_index < columns / 256; block_index++) {
        device const uchar *block = row_data + block_index * 210;
        const float delta = load_f16(block + 208);
        const uint block_base = block_index * 256;
        for (uint half_index = 0; half_index < 2; half_index++) {
            device const uchar *low = block + half_index * 64;
            device const uchar *high = block + 128 + half_index * 32;
            device const uchar *scales = block + 192 + half_index * 8;
            const uint half_base = block_base + half_index * 128;
            for (uint index = 0; index < 32; index++) {
                const uint scale_pair = index / 16;
                const uint q1 = uint(low[index] & 0x0f) | (uint(high[index] & 0x03) << 4);
                const float scale1 = delta * float(as_type<char>(scales[scale_pair]));
                sum = fma(scale1 * (float(q1) - 32.0f), input[half_base + index], sum);
            }
            for (uint index = 0; index < 32; index++) {
                const uint scale_pair = index / 16;
                const uint q2 = uint(low[index + 32] & 0x0f)
                    | (uint((high[index] >> 2) & 0x03) << 4);
                const float scale2 = delta * float(as_type<char>(scales[scale_pair + 2]));
                sum = fma(scale2 * (float(q2) - 32.0f), input[half_base + index + 32], sum);
            }
            for (uint index = 0; index < 32; index++) {
                const uint scale_pair = index / 16;
                const uint q3 = uint(low[index] >> 4)
                    | (uint((high[index] >> 4) & 0x03) << 4);
                const float scale3 = delta * float(as_type<char>(scales[scale_pair + 4]));
                sum = fma(scale3 * (float(q3) - 32.0f), input[half_base + index + 64], sum);
            }
            for (uint index = 0; index < 32; index++) {
                const uint scale_pair = index / 16;
                const uint q4 = uint(low[index + 32] >> 4)
                    | (uint((high[index] >> 6) & 0x03) << 4);
                const float scale4 = delta * float(as_type<char>(scales[scale_pair + 6]));
                sum = fma(scale4 * (float(q4) - 32.0f), input[half_base + index + 96], sum);
            }
        }
    }
    output[row] = sum;
}

kernel void attention_scores(
    device const float *query [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint &sequence_length [[buffer(3)]],
    constant uint &cache_width [[buffer(4)]],
    constant uint &query_heads [[buffer(5)]],
    constant uint &kv_heads [[buffer(6)]],
    constant uint &head_dimension [[buffer(7)]],
    constant uint &start [[buffer(8)]],
    constant uint &cache_capacity [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    const uint visible = sequence_length - start;
    const uint query_head = index / visible;
    const uint score_index = index % visible;
    const uint kv_head = query_head / (query_heads / kv_heads);
    const uint position = start + score_index;
    const uint cache_position = position % cache_capacity;
    const uint query_base = query_head * head_dimension;
    const uint key_base = cache_position * cache_width + kv_head * head_dimension;
    float sum = 0.0f;
    for (uint dimension = 0; dimension < head_dimension; dimension++) {
        sum = fma(query[query_base + dimension], keys[key_base + dimension], sum);
    }
    scores[index] = sum;
}

kernel void attention_softmax(
    device float *scores [[buffer(0)]],
    constant uint &visible [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    const uint base = head * visible;
    float maximum = -INFINITY;
    for (uint index = lane; index < visible; index += lanes) {
        maximum = max(maximum, scores[base + index]);
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
    for (uint index = lane; index < visible; index += lanes) {
        const float probability = exp(scores[base + index] - maximum);
        scores[base + index] = probability;
        sum += probability;
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
    for (uint index = lane; index < visible; index += lanes) {
        scores[base + index] *= inverse_sum;
    }
}

kernel void attention_values(
    device const float *probabilities [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &visible [[buffer(3)]],
    constant uint &start [[buffer(4)]],
    constant uint &cache_width [[buffer(5)]],
    constant uint &query_heads [[buffer(6)]],
    constant uint &kv_heads [[buffer(7)]],
    constant uint &head_dimension [[buffer(8)]],
    constant uint &cache_capacity [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    const uint query_head = index / head_dimension;
    const uint dimension = index % head_dimension;
    const uint kv_head = query_head / (query_heads / kv_heads);
    float sum = 0.0f;
    for (uint score_index = 0; score_index < visible; score_index++) {
        const uint position = start + score_index;
        const uint cache_position = position % cache_capacity;
        const uint value_index = cache_position * cache_width
            + kv_head * head_dimension + dimension;
        sum = fma(values[value_index], probabilities[query_head * visible + score_index], sum);
    }
    output[index] = sum;
}

kernel void top_k_softmax(
    device const float *logits [[buffer(0)]],
    device uint *indices [[buffer(1)]],
    device float *probabilities [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &k [[buffer(4)]]) {
    for (uint selected = 0; selected < k; selected++) {
        float best_value = -INFINITY;
        uint best_index = 0xffffffffu;
        for (uint index = 0; index < count; index++) {
            bool already_selected = false;
            for (uint prior = 0; prior < selected; prior++) {
                already_selected = already_selected || indices[prior] == index;
            }
            const float candidate = logits[index];
            if (!already_selected
                && (candidate > best_value
                    || (candidate == best_value && index < best_index))) {
                best_value = candidate;
                best_index = index;
            }
        }
        indices[selected] = best_index;
        probabilities[selected] = best_value;
    }
    const float maximum = probabilities[0];
    float sum = 0.0f;
    for (uint selected = 0; selected < k; selected++) {
        probabilities[selected] = exp(probabilities[selected] - maximum);
        sum += probabilities[selected];
    }
    for (uint selected = 0; selected < k; selected++) {
        probabilities[selected] /= sum;
    }
}
