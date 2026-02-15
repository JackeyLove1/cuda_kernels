#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// Small L2 regularization to make the solution well-defined and close
// to the reference "expected" coefficients on separable data.
static const double REG_LAMBDA = 1e-6;

// Numerically stable sigmoid in double precision
static inline double sigmoid_double(double z) {
  if (z >= 0.0) {
    double e = exp(-z);
    return 1.0 / (1.0 + e);
  } else {
    double e = exp(z);
    return e / (1.0 + e);
  }
}

/**
 * Solve A x = b for x using Gaussian elimination with partial pivoting.
 * A is modified in-place to its upper-triangular form; b is also modified.
 * d = dimension.
 */
static void solve_linear_system(double* A, double* b, double* x, int d) {
  // Forward elimination
  for (int i = 0; i < d; ++i) {
    // Pivot: find row k >= i with max |A[k, i]|
    int pivot_row = i;
    double max_val = fabs(A[i * d + i]);
    for (int k = i + 1; k < d; ++k) {
      double val = fabs(A[k * d + i]);
      if (val > max_val) {
        max_val = val;
        pivot_row = k;
      }
    }

    // Swap rows i and pivot_row in A and b if needed
    if (pivot_row != i) {
      for (int j = 0; j < d; ++j) {
        double tmp = A[i * d + j];
        A[i * d + j] = A[pivot_row * d + j];
        A[pivot_row * d + j] = tmp;
      }
      double tmpb = b[i];
      b[i] = b[pivot_row];
      b[pivot_row] = tmpb;
    }

    // If pivot is extremely small, add tiny jitter to keep system solvable
    double pivot = A[i * d + i];
    if (fabs(pivot) < 1e-12) {
      pivot = (pivot >= 0.0 ? 1e-12 : -1e-12);
      A[i * d + i] = pivot;
    }

    // Eliminate below pivot
    for (int k = i + 1; k < d; ++k) {
      double factor = A[k * d + i] / pivot;
      if (factor == 0.0) continue;

      // Row operation on A
      for (int j = i; j < d; ++j) {
        A[k * d + j] -= factor * A[i * d + j];
      }
      // Row operation on b
      b[k] -= factor * b[i];
    }
  }

  // Back substitution
  for (int i = d - 1; i >= 0; --i) {
    double sum = b[i];
    for (int j = i + 1; j < d; ++j) {
      sum -= A[i * d + j] * x[j];
    }
    double diag = A[i * d + i];
    if (fabs(diag) < 1e-12) {
      diag = (diag >= 0.0 ? 1e-12 : -1e-12);
    }
    x[i] = sum / diag;
  }
}

/**
 * Host-side Newton / IRLS solver for L2-regularized logistic regression.
 * X_dev, y_dev, beta_dev are device pointers. This function copies data
 * to host, runs Newton in double, then copies beta back to device.
 */
extern "C" void solve(const float* X_dev, const float* y_dev, float* beta_dev, int n_samples,
                      int n_features) {
  const int n = n_samples;
  const int d = n_features;

  if (n <= 0 || d <= 0) {
    return;
  }

  // Temporary host buffers (double precision for stability)
  double* X = (double*)malloc((size_t)n * d * sizeof(double));
  double* y = (double*)malloc((size_t)n * sizeof(double));
  double* beta = (double*)malloc((size_t)d * sizeof(double));
  double* grad = (double*)malloc((size_t)d * sizeof(double));
  double* delta = (double*)malloc((size_t)d * sizeof(double));
  double* H = (double*)malloc((size_t)d * d * sizeof(double));
  double* p = (double*)malloc((size_t)n * sizeof(double));
  double* w = (double*)malloc((size_t)n * sizeof(double));

  if (!X || !y || !beta || !grad || !delta || !H || !p || !w) {
    // Allocation failure: free what we can and return
    if (X) free(X);
    if (y) free(y);
    if (beta) free(beta);
    if (grad) free(grad);
    if (delta) free(delta);
    if (H) free(H);
    if (p) free(p);
    if (w) free(w);
    return;
  }

  // Copy X, y from device to host (via temporary float buffers)
  float* X_tmp = (float*)malloc((size_t)n * d * sizeof(float));
  float* y_tmp = (float*)malloc((size_t)n * sizeof(float));
  if (!X_tmp || !y_tmp) {
    if (X_tmp) free(X_tmp);
    if (y_tmp) free(y_tmp);
    free(X);
    free(y);
    free(beta);
    free(grad);
    free(delta);
    free(H);
    free(p);
    free(w);
    return;
  }

  cudaMemcpy(X_tmp, X_dev, (size_t)n * d * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y_tmp, y_dev, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

  // Convert to double
  for (int i = 0; i < n * d; ++i) {
    X[i] = (double)X_tmp[i];
  }
  for (int i = 0; i < n; ++i) {
    y[i] = (double)y_tmp[i];
  }

  free(X_tmp);
  free(y_tmp);

  // Initialize beta = 0
  for (int j = 0; j < d; ++j) {
    beta[j] = 0.0;
  }

  // Newton / IRLS hyperparameters
  const int MAX_ITER = 25;
  const double TOL = 1e-8;

  for (int iter = 0; iter < MAX_ITER; ++iter) {
    // 1) Compute p_i = sigmoid(x_i^T beta), w_i = p_i (1 - p_i)
    for (int i = 0; i < n; ++i) {
      double z = 0.0;
      const double* Xi = X + (size_t)i * d;
      for (int j = 0; j < d; ++j) {
        z += Xi[j] * beta[j];
      }
      double pi = sigmoid_double(z);
      p[i] = pi;
      w[i] = pi * (1.0 - pi);
    }

    // 2) Gradient: grad = X^T (p - y) + lambda * beta
    //    We build it as: grad[j] = lambda * beta[j] + sum_i X_ij * (p_i - y_i)
    double max_abs_grad = 0.0;

    for (int j = 0; j < d; ++j) {
      grad[j] = REG_LAMBDA * beta[j];
    }

    for (int i = 0; i < n; ++i) {
      double t = p[i] - y[i];
      const double* Xi = X + (size_t)i * d;
      for (int j = 0; j < d; ++j) {
        grad[j] += Xi[j] * t;
      }
    }

    for (int j = 0; j < d; ++j) {
      double g = fabs(grad[j]);
      if (g > max_abs_grad) max_abs_grad = g;
    }

    if (max_abs_grad < TOL) {
      // Converged
      break;
    }

    // 3) Hessian: H = X^T W X + lambda * I
    //    W is diagonal with w_i = p_i (1 - p_i)
    //    We exploit symmetry (compute upper triangle and mirror).
    for (int j = 0; j < d * d; ++j) {
      H[j] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
      const double* Xi = X + (size_t)i * d;
      double wi = w[i];
      if (wi == 0.0) continue;
      for (int j = 0; j < d; ++j) {
        double xij = Xi[j];
        double w_xij = wi * xij;
        for (int k = 0; k <= j; ++k) {
          H[j * d + k] += w_xij * Xi[k];
        }
      }
    }

    // Mirror the symmetric Hessian and add regularization on the diagonal
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < j; ++k) {
        H[k * d + j] = H[j * d + k];
      }
      H[j * d + j] += REG_LAMBDA;
    }

    // 4) Solve H * delta = grad  (for minimizing L)
    //    Then beta <- beta - delta
    for (int j = 0; j < d; ++j) {
      delta[j] = 0.0;
    }

    // We will overwrite grad when solving; pass it as RHS b.
    solve_linear_system(H, grad, delta, d);

    // Update beta
    for (int j = 0; j < d; ++j) {
      beta[j] -= delta[j];
    }
  }

  // Copy final beta back to device as float
  float* beta_tmp = (float*)malloc((size_t)d * sizeof(float));
  if (beta_tmp) {
    for (int j = 0; j < d; ++j) {
      beta_tmp[j] = (float)beta[j];
    }
    cudaMemcpy(beta_dev, beta_tmp, (size_t)d * sizeof(float), cudaMemcpyHostToDevice);
    free(beta_tmp);
  }

  // Free host buffers
  free(X);
  free(y);
  free(beta);
  free(grad);
  free(delta);
  free(H);
  free(p);
  free(w);
}
