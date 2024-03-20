#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

void dot_product(const float* A, 
                 const float* B, 
                 float* C, 
                 size_t A_rows, 
                 size_t A_cols, 
                 size_t B_cols) {
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] /= sum_exp;
    }

    return probabilities;
}

// Function to compute the gradient of the softmax loss
void softmax_grad(const std::vector<float>& Z, 
                  const std::vector<unsigned char>& y,
                  size_t batch_size, 
                  size_t num_classes, 
                  std::vector<float>& gradient) {
    std::vector<float> probabilities;

    // Compute softmax probabilities
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> logits(Z.begin() + i * num_classes, Z.begin() + (i + 1) * num_classes);
        std::vector<float> probs = softmax(logits);

        // Subtract 1 from the probability of the correct class
        probs[y[i]] -= 1;

        // Copy the probabilities back into the gradient vector
        std::copy(probs.begin(), probs.end(), gradient.begin() + i * num_classes);
    }


    // Average the gradient over the batch
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient[i] /= static_cast<float>(batch_size);
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);
        std::vector<float> logits_vec(current_batch_size * k);
        // 计算当前批次的logits
        dot_product(X + i * n, theta, logits_vec.data(), current_batch_size, n, k);
        std::vector<float> gradient_vec(current_batch_size * k);
        // 调用softmax_grad函数计算梯度
        softmax_grad(logits_vec, std::vector<unsigned char>(y + i, y + i + current_batch_size),
                     current_batch_size, k, gradient_vec);
        // 对于当前批次中的每个样本（由外循环j控制）
        for (size_t j = 0; j < current_batch_size; ++j) {
            // 对于每个类别（由中循环c控制）
            for (size_t c = 0; c < k; ++c) {
                // 对于每个特征（由内循环d控制）
                for (size_t d = 0; d < n; ++d) {
                    theta[d * k + c] -= lr * gradient_vec[j * k + c] * X[(i + j) * n + d];
                }
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
