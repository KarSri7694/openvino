// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <algorithm>
#include <limits>
#include <memory>

namespace ov::intel_gpu {

MultiTensorState::MultiTensorState(const std::vector<VariableStateInfo>& infos,
                                   std::shared_ptr<RemoteContextImpl> context,
                                   ShapePredictor::Ptr shape_predictor) : ov::intel_gpu::VariableStateBase(infos[0].m_id, context) {
    for (auto& info : infos) {
        m_hidden_states.push_back(std::make_shared<VariableState>(info, context, shape_predictor));
    }
}

VariableStateIndirectKVCache::VariableStateIndirectKVCache(const VariableStateInfo& info,
                                                           RemoteContextImpl::Ptr context,
                                                           std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                                           size_t beam_axis,
                                                           size_t concat_axis)
    : MultiTensorState { {info}, context, shape_predictor}
    , m_beam_axis(beam_axis)
    , m_concat_axis(concat_axis) {
    cldnn::layout beam_table_layout(get_beam_table_shape(info.m_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    VariableStateInfo beam_table_state_info(info.m_id + "/beam_table", beam_table_layout);
    beam_table_state_info.m_release_variable_inst = info.m_release_variable_inst;
    m_hidden_states.push_back(std::make_shared<VariableState>(beam_table_state_info, context, shape_predictor));
    OPENVINO_ASSERT(m_hidden_states.size() == 2, "[GPU] VariableStateIndirectKVCache expects 2 internal states to be initialized");
}

void VariableStateIndirectKVCache::reset() {
    for (auto& state : m_hidden_states) {
        state->reset();
    }
    m_is_set = false;
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_memory() const {
    return m_hidden_states[0]->get_memory();
}

const cldnn::layout& VariableStateIndirectKVCache::get_layout() const {
    return m_hidden_states[0]->get_layout();
}

void VariableStateIndirectKVCache::set_state(const ov::SoPtr<ov::ITensor>& state) {
    OPENVINO_ASSERT(m_hidden_states.size() == 2, "[GPU] Corrupted VariableStateIndirectKVCache. Expected 2 internal states. Got: ", m_hidden_states.size());
    m_hidden_states[0]->set_state(state); // user can set only KV cache

    // Beam table is reset to cleanup rearranges history
    cldnn::layout bt_layout(get_beam_table_shape(state->get_shape()), ov::element::i32, cldnn::format::bfyx);
    m_hidden_states[1]->reset();
    m_hidden_states[1]->set_layout(bt_layout);
}

template<typename T>
void copy_element(const void* src, void* dst, size_t src_offset, size_t dst_offset) {
    static_cast<T*>(dst)[dst_offset] = static_cast<const T*>(src)[src_offset];
}

static void rearrange_cache(cldnn::memory::ptr kv_in_mem, cldnn::memory::ptr bt_mem, cldnn::memory::ptr kv_out_mem, cldnn::stream& stream, size_t concat_axis) {
    auto kv_layout = kv_in_mem->get_layout();
    auto kv_shape = kv_layout.get_shape();
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> kv_in_ptr(kv_in_mem, stream);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> bt_in_ptr(bt_mem, stream);
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> kv_out_ptr(kv_out_mem, stream);

    OPENVINO_ASSERT(kv_shape.size() == 4);

    for (size_t b = 0; b < kv_shape[0]; b++) {
        for (size_t f = 0; f < kv_shape[1]; f++) {
            for (size_t y = 0; y < kv_shape[2]; y++) {
                for (size_t x = 0; x < kv_shape[3]; x++) {
                    auto out_idx = std::vector<ov::Dimension::value_type>{static_cast<ov::Dimension::value_type>(b),
                                                                          static_cast<ov::Dimension::value_type>(f),
                                                                          static_cast<ov::Dimension::value_type>(y),
                                                                          static_cast<ov::Dimension::value_type>(x)};

                    size_t b_kv = bt_in_ptr[b * kv_shape[concat_axis] + out_idx[concat_axis]]; // bt_idx = b * total_seq_len + seq_len_idx
                    auto in_idx = std::vector<ov::Dimension::value_type>{static_cast<ov::Dimension::value_type>(b_kv),
                                                                         static_cast<ov::Dimension::value_type>(f),
                                                                         static_cast<ov::Dimension::value_type>(y),
                                                                         static_cast<ov::Dimension::value_type>(x)};

                    cldnn::tensor in(cldnn::format::bfyx, in_idx, static_cast<ov::Dimension::value_type>(0));
                    cldnn::tensor out(cldnn::format::bfyx, out_idx, static_cast<ov::Dimension::value_type>(0));

                    size_t out_offset = kv_out_mem->get_layout().get_linear_offset(out);
                    size_t in_offset = kv_layout.get_linear_offset(in);

                    if (ov::element::Type(kv_layout.data_type).size() == 2)
                        copy_element<uint16_t>(kv_in_ptr.data(), kv_out_ptr.data(), in_offset, out_offset);
                    else if (ov::element::Type(kv_layout.data_type).size() == 4)
                        copy_element<uint32_t>(kv_in_ptr.data(), kv_out_ptr.data(), in_offset, out_offset);
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCache::get_state() const {
    auto kv_layout = m_hidden_states[0]->get_layout();
    auto bt_mem = m_hidden_states[1]->get_memory();
    if (kv_layout.get_partial_shape()[m_beam_axis].get_length() > 1 && bt_mem) {
        auto kv_mem = m_hidden_states[0]->get_memory();
        auto tensor = m_context->create_host_tensor(m_hidden_states[0]->get_user_specified_type(), kv_layout.get_shape());

        auto& engine = m_context->get_engine();
        auto tmp_mem = engine.allocate_memory(kv_layout, engine.get_lockable_preferred_memory_allocation_type(), false);

        rearrange_cache(kv_mem, bt_mem, tmp_mem, m_context->get_engine().get_service_stream(), m_concat_axis);

        convert_and_copy(tmp_mem, tensor._ptr.get(), m_context->get_engine().get_service_stream());

        return tensor;
    } else {
        return m_hidden_states[0]->get_state();
    }
}

void VariableStateIndirectKVCache::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    m_hidden_states[0]->set_memory(new_mem, actual_layout);
}

void VariableStateIndirectKVCache::set_layout(const cldnn::layout& new_layout) {
    m_hidden_states[0]->set_layout(new_layout);
}

size_t VariableStateIndirectKVCache::get_actual_mem_size() const {
    return m_hidden_states[0]->get_actual_mem_size();
}

ov::PartialShape VariableStateIndirectKVCache::get_beam_table_shape(const ov::PartialShape& kv_cache_shape) {
    auto rank = kv_cache_shape.size();
    ov::PartialShape beam_table_shape(std::vector<size_t>(rank, 1));
    beam_table_shape[m_beam_axis] = kv_cache_shape[m_beam_axis];
    beam_table_shape[m_concat_axis] = kv_cache_shape[m_concat_axis];
    return beam_table_shape;
}

VariableState::Ptr VariableStateIndirectKVCache::get_beam_table_state() const {
    return m_hidden_states[1];
}

VariableStateIndirectKVCacheCompressed::VariableStateIndirectKVCacheCompressed(
    const VariableStateInfo& info,
    std::shared_ptr<RemoteContextImpl> context,
    std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
    const std::vector<cldnn::layout>& output_layouts,
    size_t beam_idx,
    size_t concat_idx,
    bool has_zp_state = false)
    : VariableStateIndirectKVCache(info, context, shape_predictor, beam_idx, concat_idx),
      m_has_zp_state(has_zp_state) {
    OPENVINO_ASSERT((has_zp_state && output_layouts.size() == 3) ||
                    (!has_zp_state && output_layouts.size() == 2),
                    "[GPU] Unexpected number of output layouts for VariableStateIndirectKVCacheCompressed");

    const auto compression_scale_layout = output_layouts[1];
    VariableStateInfo compression_scale_state_info(info.m_id + "/comp_scale", compression_scale_layout);
    m_hidden_states.push_back(std::make_shared<VariableState>(compression_scale_state_info, context, shape_predictor));

    if (has_zp_state) {
        const auto compression_zp_layout = output_layouts[2];
        VariableStateInfo compression_zp_state_info(info.m_id + "/comp_zp", compression_zp_layout);
        m_hidden_states.push_back(std::make_shared<VariableState>(compression_zp_state_info, context, shape_predictor));
    }

    OPENVINO_ASSERT((!m_has_zp_state && m_hidden_states.size() == 3) || (m_has_zp_state && m_hidden_states.size() == 4),
                    "[GPU] VariableStateIndirectKVCacheCompressed expects 3 or 4 internal states to be initialized, "
                    "actual number is ", m_hidden_states.size());
}

VariableState::Ptr VariableStateIndirectKVCacheCompressed::get_compression_scale_state() const {
    return m_hidden_states[2];
}

void VariableStateIndirectKVCacheCompressed::set_compression_scale_layout(const cldnn::layout& new_layout) {
    m_hidden_states[2]->set_layout(new_layout);
}

VariableState::Ptr VariableStateIndirectKVCacheCompressed::get_compression_zp_state() const {
    OPENVINO_ASSERT(m_has_zp_state);
    return m_hidden_states[3];
}

void VariableStateIndirectKVCacheCompressed::set_compression_zp_layout(const cldnn::layout& new_layout) {
    OPENVINO_ASSERT(m_has_zp_state);
    m_hidden_states[3]->set_layout(new_layout);
}

bool VariableStateIndirectKVCacheCompressed::has_zp_state() const {
    return m_has_zp_state;
}

void VariableStateIndirectKVCacheCompressed::set_state(const ov::SoPtr<ov::ITensor>& state) {
    const auto src_shape = state->get_shape();
    const auto src_type = state->get_element_type();
    OPENVINO_ASSERT(src_shape.size() == 4,
                    "[GPU] VariableStateIndirectKVCacheCompressed::set_state: expected 4D tensor, got rank ", src_shape.size());
    OPENVINO_ASSERT(src_type == ov::element::f32 || src_type == ov::element::f16,
                    "[GPU] VariableStateIndirectKVCacheCompressed::set_state: expected f32 or f16 input, got ",
                    src_type.get_type_name());

    const size_t B = src_shape[0], H = src_shape[1], S = src_shape[2], D = src_shape[3];
    auto& engine = m_context->get_engine();
    auto& stream = engine.get_service_stream();

    // Helper: read one element from the (possibly f16 or f32, possibly non-contiguous) source tensor as f32
    const bool src_is_f16 = (src_type == ov::element::f16);
    auto read_f32 = [&](size_t b, size_t h, size_t s, size_t d) -> float {
        const uint8_t* raw = static_cast<const uint8_t*>(state->data());
        const auto& strides = state->get_strides();  // bytes per element in each dim
        const size_t off = b * strides[0] + h * strides[1] + s * strides[2] + d * strides[3];
        return src_is_f16 ? static_cast<float>(*reinterpret_cast<const ov::float16*>(raw + off))
                          : *reinterpret_cast<const float*>(raw + off);
    };

    // Build new compressed layouts, preserving dtype/format from the original initial layouts
    const auto& kv_template    = m_hidden_states[0]->get_initial_layout();
    const auto& scale_template = m_hidden_states[2]->get_initial_layout();
    // Use get_partial_shape() here — get_initial_layout() has a dynamic sequence dimension,
    // so calling get_shape() (which calls to_shape()) would crash.
    // The last dim (group size / interleaved-ZP flag) must be static.
    const auto scale_pshape = scale_template.get_partial_shape();
    OPENVINO_ASSERT(scale_pshape.rank().is_static() && scale_pshape.rank().get_length() == 4,
                    "[GPU] VariableStateIndirectKVCacheCompressed::set_state: expected rank-4 compression scale shape");
    OPENVINO_ASSERT(scale_pshape[3].is_static(),
                    "[GPU] VariableStateIndirectKVCacheCompressed::set_state: expected static last dim for compression scale");
    const size_t scale_last_dim = scale_pshape[3].get_length();
    const bool has_interleaved_zp = !m_has_zp_state && scale_last_dim > 1;

    const cldnn::layout new_kv_layout(ov::PartialShape(src_shape),
                                      kv_template.data_type, kv_template.format);
    const cldnn::layout new_scale_layout(ov::PartialShape(ov::Shape{B, H, S, scale_last_dim}),
                                         scale_template.data_type, scale_template.format);

    cldnn::memory::ptr new_zp_mem;
    cldnn::layout new_zp_layout;
    if (m_has_zp_state) {
        const auto& zp_template = m_hidden_states[3]->get_initial_layout();
        new_zp_layout = cldnn::layout(ov::PartialShape(ov::Shape{B, H, S, 1}),
                                      zp_template.data_type, zp_template.format);
    }

    const auto alloc_type = engine.get_lockable_preferred_memory_allocation_type();
    auto new_kv_mem    = engine.allocate_memory(new_kv_layout,    alloc_type, false);
    auto new_scale_mem = engine.allocate_memory(new_scale_layout, alloc_type, false);
    if (m_has_zp_state)
        new_zp_mem = engine.allocate_memory(new_zp_layout, alloc_type, false);

    // Quantize per token to match GPU kernels:
    // - Asymmetric: q = round(v * (255/diff) + zp), dequant = (q - zp) * (diff/255)
    // - Symmetric:  q = round(v * (127/max_abs)),   dequant = q * (max_abs/127)
    {
        cldnn::mem_lock<int8_t,       cldnn::mem_lock_type::write> kv_out(new_kv_mem,    stream);
        cldnn::mem_lock<ov::float16,  cldnn::mem_lock_type::write> scale_out(new_scale_mem, stream);
        std::unique_ptr<cldnn::mem_lock<int8_t, cldnn::mem_lock_type::write>> zp_out;
        if (m_has_zp_state)
            zp_out = std::make_unique<cldnn::mem_lock<int8_t, cldnn::mem_lock_type::write>>(new_zp_mem, stream);

        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t s = 0; s < S; s++) {
                    const std::vector<ov::Dimension::value_type> scale_idx{(int)b, (int)h, (int)s, 0};
                    cldnn::tensor scale_t(cldnn::format::bfyx, scale_idx, 0);

                    if (!m_has_zp_state && !has_interleaved_zp) {
                        // Symmetric mode (no ZP tensor, no interleaved ZP)
                        float max_abs = std::abs(read_f32(b, h, s, 0));
                        for (size_t d = 1; d < D; d++) {
                            max_abs = std::max(max_abs, std::abs(read_f32(b, h, s, d)));
                        }
                        max_abs = std::max(max_abs, 0.001f);

                        const float quant_scale = 127.0f / max_abs;
                        const float stored_scale = 1.0f / quant_scale;
                        scale_out[new_scale_layout.get_linear_offset(scale_t)] = ov::float16(stored_scale);

                        for (size_t d = 0; d < D; d++) {
                            float q = std::round(read_f32(b, h, s, d) * quant_scale);
                            q = std::max(static_cast<float>(std::numeric_limits<int8_t>::min()),
                                         std::min(static_cast<float>(std::numeric_limits<int8_t>::max()), q));
                            const std::vector<ov::Dimension::value_type> kv_idx{(int)b, (int)h, (int)s, (int)d};
                            cldnn::tensor kv_t(cldnn::format::bfyx, kv_idx, 0);
                            kv_out[new_kv_layout.get_linear_offset(kv_t)] = static_cast<int8_t>(q);
                        }
                    } else {
                        // Asymmetric mode (separate ZP tensor or interleaved ZP)
                        float min_val = read_f32(b, h, s, 0);
                        float max_val = min_val;
                        for (size_t d = 1; d < D; d++) {
                            const float v = read_f32(b, h, s, d);
                            min_val = std::min(min_val, v);
                            max_val = std::max(max_val, v);
                        }

                        float diff = max_val - min_val;
                        diff = diff == 0.0f ? 0.001f : diff;

                        const float stored_scale = diff / 255.0f;
                        const float quant_scale  = 255.0f / diff;
                        const float stored_zp    = -min_val * quant_scale
                                                   + static_cast<float>(std::numeric_limits<int8_t>::min());

                        scale_out[new_scale_layout.get_linear_offset(scale_t)] = ov::float16(stored_scale);

                        if (m_has_zp_state) {
                            // Planar mode: ZP stored as i8 in a separate tensor
                            const float zp_clamped = std::max(static_cast<float>(std::numeric_limits<int8_t>::min()),
                                                              std::min(static_cast<float>(std::numeric_limits<int8_t>::max()),
                                                                       std::round(stored_zp)));
                            (*zp_out)[new_zp_layout.get_linear_offset(scale_t)] = static_cast<int8_t>(zp_clamped);
                        } else {
                            // Interleaved mode: ZP packed as f16 at [b, h, s, 1] in the scale tensor
                            const std::vector<ov::Dimension::value_type> zp_idx{(int)b, (int)h, (int)s, 1};
                            cldnn::tensor zp_t(cldnn::format::bfyx, zp_idx, 0);
                            scale_out[new_scale_layout.get_linear_offset(zp_t)] = ov::float16(stored_zp);
                        }

                        for (size_t d = 0; d < D; d++) {
                            float q = std::round(read_f32(b, h, s, d) * quant_scale + stored_zp);
                            q = std::max(static_cast<float>(std::numeric_limits<int8_t>::min()),
                                         std::min(static_cast<float>(std::numeric_limits<int8_t>::max()), q));
                            const std::vector<ov::Dimension::value_type> kv_idx{(int)b, (int)h, (int)s, (int)d};
                            cldnn::tensor kv_t(cldnn::format::bfyx, kv_idx, 0);
                            kv_out[new_kv_layout.get_linear_offset(kv_t)] = static_cast<int8_t>(q);
                        }
                    }
                }
            }
        }
    }  // locks released — writes flushed before set_memory

    m_hidden_states[0]->set_memory(new_kv_mem, new_kv_layout);
    m_hidden_states[0]->set();
    m_hidden_states[2]->set_memory(new_scale_mem, new_scale_layout);
    m_hidden_states[2]->set();
    if (m_has_zp_state) {
        m_hidden_states[3]->set_memory(new_zp_mem, new_zp_layout);
        m_hidden_states[3]->set();
    }

    // Reset beam table to identity for the new sequence
    const cldnn::layout bt_layout(get_beam_table_shape(src_shape), ov::element::i32, cldnn::format::bfyx);
    m_hidden_states[1]->reset();
    m_hidden_states[1]->set_layout(bt_layout);
    m_is_set = true;
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCacheCompressed::get_state() const {
    auto kv_mem = m_hidden_states[0]->get_memory();
    if (!kv_mem)
        return m_hidden_states[0]->get_state();  // uninitialized: delegate for empty-tensor handling

    auto scale_mem = m_hidden_states[2]->get_memory();
    OPENVINO_ASSERT(scale_mem, "[GPU] VariableStateIndirectKVCacheCompressed: compression scale memory not initialised");

    const auto kv_layout    = m_hidden_states[0]->get_layout();
    const auto scale_layout = m_hidden_states[2]->get_layout();
    const auto kv_shape     = kv_layout.get_shape();
    OPENVINO_ASSERT(kv_shape.size() == 4, "[GPU] VariableStateIndirectKVCacheCompressed: expected 4D KV cache");
    const auto scale_pshape = scale_layout.get_partial_shape();
    OPENVINO_ASSERT(scale_pshape.rank().is_static() && scale_pshape.rank().get_length() == 4,
                    "[GPU] VariableStateIndirectKVCacheCompressed::get_state: expected rank-4 compression scale shape");
    OPENVINO_ASSERT(scale_pshape[3].is_static(),
                    "[GPU] VariableStateIndirectKVCacheCompressed::get_state: expected static last dim for compression scale");
    const size_t scale_last_dim = scale_pshape[3].get_length();
    const bool has_interleaved_zp = !m_has_zp_state && scale_last_dim > 1;

    auto& stream = m_context->get_engine().get_service_stream();

    // Output in the scale tensor's dtype (f16 — the original computation precision)
    const auto out_type = ov::element::Type(scale_layout.data_type);
    auto tensor = m_context->create_host_tensor(out_type, kv_shape);

    const auto bt_mem   = m_hidden_states[1]->get_memory();
    const bool use_beam = kv_layout.get_partial_shape()[m_beam_axis].get_length() > 1 && bt_mem;

    cldnn::mem_lock<int8_t,      cldnn::mem_lock_type::read> kv_in_ptr(kv_mem,    stream);
    cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> scale_ptr(scale_mem, stream);
    std::unique_ptr<cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read>> bt_in_ptr;
    if (use_beam)
        bt_in_ptr = std::make_unique<cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read>>(bt_mem, stream);

    // Write a dequantised float value to the (contiguous) output host tensor
    auto write_out = [&](float val, size_t b, size_t f, size_t y, size_t x) {
        const size_t off = (b * kv_shape[1] + f) * kv_shape[2] * kv_shape[3] + y * kv_shape[3] + x;
        if (out_type == ov::element::f16)
            static_cast<ov::float16*>(tensor->data())[off] = ov::float16(val);
        else
            static_cast<float*>(tensor->data())[off] = val;
    };

    // Inner loop shared by both storage modes; get_zp(b_kv, f, y) returns the f32 zero-point
    auto run_dequantize = [&](auto get_zp) {
        for (size_t b = 0; b < kv_shape[0]; b++) {
            for (size_t f = 0; f < kv_shape[1]; f++) {
                for (size_t y = 0; y < kv_shape[2]; y++) {
                    const std::vector<ov::Dimension::value_type> out_idx{(int)b, (int)f, (int)y, 0};
                    const size_t b_kv = use_beam
                        ? static_cast<size_t>((*bt_in_ptr)[b * kv_shape[m_concat_axis] + out_idx[m_concat_axis]])
                        : b;

                    const std::vector<ov::Dimension::value_type> src_idx{(int)b_kv, (int)f, (int)y, 0};
                    cldnn::tensor scale_t(cldnn::format::bfyx, src_idx, 0);
                    const float scale = static_cast<float>(scale_ptr[scale_layout.get_linear_offset(scale_t)]);
                    const float zp    = get_zp(b_kv, f, y, src_idx);

                    for (size_t x = 0; x < kv_shape[3]; x++) {
                        const std::vector<ov::Dimension::value_type> in_idx{(int)b_kv, (int)f, (int)y, (int)x};
                        cldnn::tensor in_t(cldnn::format::bfyx, in_idx, 0);
                        const float kv_val = static_cast<float>(kv_in_ptr[kv_layout.get_linear_offset(in_t)]);
                        write_out((kv_val - zp) * scale, b, f, y, x);
                    }
                }
            }
        }
    };

    if (m_has_zp_state) {
        // Planar mode: separate i8 ZP tensor
        const auto zp_mem    = m_hidden_states[3]->get_memory();
        const auto zp_layout = m_hidden_states[3]->get_layout();
        cldnn::mem_lock<int8_t, cldnn::mem_lock_type::read> zp_ptr(zp_mem, stream);
        run_dequantize([&](size_t, size_t, size_t,
                           const std::vector<ov::Dimension::value_type>& idx) -> float {
            cldnn::tensor t(cldnn::format::bfyx, idx, 0);
            return static_cast<float>(zp_ptr[zp_layout.get_linear_offset(t)]);
        });
    } else if (has_interleaved_zp) {
        // Interleaved mode: ZP packed at [b_kv, f, y, 1] in the scale tensor
        run_dequantize([&](size_t b_kv, size_t f, size_t y,
                           const std::vector<ov::Dimension::value_type>&) -> float {
            const std::vector<ov::Dimension::value_type> zp_idx{(int)b_kv, (int)f, (int)y, 1};
            cldnn::tensor zp_t(cldnn::format::bfyx, zp_idx, 0);
            return static_cast<float>(scale_ptr[scale_layout.get_linear_offset(zp_t)]);
        });
    } else {
        // Symmetric mode: no zero-point is stored
        run_dequantize([&](size_t, size_t, size_t,
                           const std::vector<ov::Dimension::value_type>&) -> float {
            return 0.0f;
        });
    }

    return tensor;
}

}  // namespace ov::intel_gpu   
