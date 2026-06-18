// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_x4x2_tensor.h
 * @date	28 May 2026
 * @brief	Q4_0_X4X2_Tensor: Q4_0 weights stored in DSP-native x4x2 layout.
 *              Byte size identical to Q4_0_Tensor; only the reported data-type
 *              string differs so the runtime dispatch can skip the x4x2 repack.
 * @see		https://github.com/nntrainer/nntrainer
 */

#ifndef __Q4_0_X4X2_TENSOR_H__
#define __Q4_0_X4X2_TENSOR_H__
#ifdef __cplusplus

#include <q4_0_tensor.h>

namespace nntrainer {

/**
 * @class Q4_0_X4X2_Tensor
 * @brief Q4_0 tensor whose bytes are pre-stored in the x4x2 row-strided
 * layout.
 */
class Q4_0_X4X2_Tensor : public Q4_0_Tensor {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Q4_0_X4X2_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    Q4_0_Tensor(name_, fm) {}

  /**
   * @brief Construct a new Q4_0_X4X2_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  Q4_0_X4X2_Tensor(const TensorDim &d, bool alloc_now,
                   Initializer init = Initializer::NONE,
                   std::string name = "") :
    Q4_0_Tensor(d, alloc_now, init, name) {}

  /**
   * @brief Construct a new Q4_0_X4X2_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  Q4_0_X4X2_Tensor(const TensorDim &d, const void *buf = nullptr) :
    Q4_0_Tensor(d, buf) {}

  /**
   * @brief Construct a new Q4_0_X4X2_Tensor object
   * @param rhs TensorBase object to copy
   */
  Q4_0_X4X2_Tensor(TensorBase &rhs) : Q4_0_Tensor(rhs) {}

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override { return QScheme::Q4_0; }

private:
  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (Q4_0_X4X2)
   */
  std::string getStringDataType() const override { return "Q4_0_X4X2"; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q4_0_X4X2_TENSOR_H__ */
