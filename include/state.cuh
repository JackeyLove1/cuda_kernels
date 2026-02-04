/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_STATE_CUH_
#define FLASHINFER_STATE_CUH_

#pragma once

#include "include/math.cuh"
#include "include/vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief The flashattention state.
 * \tparam vec_size The size of the vector used in o.
 */
template <size_t vec_size>
struct state_t {
  /* the weighted sum of v: exp(pre-softmax logit - m) * v / d  */
  vec_t<float, vec_size> o;
  /* maximum value of pre-softmax logits */
  float m;
  /* sum of exp(pre-softmax logits - m) */
  float d;

  __device__ __forceinline__ void init() {
    o.fill(0.f);
    m = -math::inf;
    d = 1.f;
  }

  __device__ __forceinline__ state_t() { init(); }

  __device__ __forceinline__ float get_lse() const { return m + math::ptx_log2(d); }

  /*!
   * \brief Merge the state with another state.
   * \param other_m The maximum value of pre-softmax logits of the other state.
   * \param other_d The sum of exp(pre-softmax logits - m) of the other state.
   * \param other_o The weighted sum of v of the other state.
   */
  __device__ __forceinline__ void merge(const vec_t<float, vec_size>& other_o, float other_m,
                                        float other_d) {
    float m_prev = m, d_prev = d;
    m = max(m_prev, other_m);
    // Compute the rescaling factors once (avoid 2x exp2 per element).
    const float scale_prev = math::ptx_exp2(m_prev - m);
    const float scale_other = math::ptx_exp2(other_m - m);
    d = d_prev * scale_prev + other_d * scale_other;

    float4* o4 = reinterpret_cast<float4*>(o.ptr());
    float4* other4 = reinterpret_cast<float4*>(const_cast<vec_t<float, vec_size>&>(other_o).ptr());
    constexpr auto vecs = vec_size / 4;
    constexpr auto remain = vec_size % 4;
#pragma unroll
      for (size_t i = 0; i < vecs; ++i) {
        float4 a = o4[i];
        const float4 b = other4[i];
        a.x = a.x * scale_prev + b.x * scale_other;
        a.y = a.y * scale_prev + b.y * scale_other;
        a.z = a.z * scale_prev + b.z * scale_other;
        a.w = a.w * scale_prev + b.w * scale_other;
        o4[i] = a;
      }
#pragma unroll
      for (size_t i = 0; i < remain; ++i) {
        o[i] = o[i] * scale_prev + other_o[i] * scale_other;
      }
  }

  /*!
   * \brief Merge the state with another state.
   * \param other The other state.
   */
  __device__ __forceinline__ void merge(const state_t<vec_size>& other) {
    merge(other.o, other.m, other.d);
  }

  __device__ __forceinline__ void normalize() {
    // only normalize by d when not normalized on the fly
    const float inv_d = __fdividef(1.f, d);

    float4* o4 = reinterpret_cast<float4*>(o.ptr());
    constexpr auto vecs = vec_size / 4;
    constexpr auto remain = vec_size % 4;
#pragma unroll
      for (size_t i = 0; i < vecs; ++i) {
        float4 a = o4[i];
        a.x *= inv_d;
        a.y *= inv_d;
        a.z *= inv_d;
        a.w *= inv_d;
        o4[i] = a;
      }
#pragma unroll
      for (size_t i = 0; i < remain; ++i) {
        o[i] *= inv_d;
      }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_STATE_CUH_
