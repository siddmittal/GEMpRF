import cupy as cp

code = r"""
__device__ int a_plus_b(const int& a, const int& b) {
    return a + b;
}

__device__ int a_minus_b(const int& a, const int& b) {
    return a - b;
}

template<auto Func>
__global__ void my_kernel(const int* a, const int* b, int* out, int N) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        out[tid] = Func(a[tid], b[tid]);
    }
}
"""

mod = cp.RawModule(code=code,
                   options=("-std=c++17",),
                   name_expressions=('my_kernel<a_plus_b>', 'my_kernel<a_minus_b>'))
plus_ker = mod.get_function('my_kernel<a_plus_b>')
minus_ker = mod.get_function('my_kernel<a_minus_b>')

a = cp.arange(5, dtype=cp.int32)
b = cp.arange(5, dtype=cp.int32) + 5
c = cp.empty_like(a)
plus_ker((1,), (5,), (a, b, c, 5))
print(f"{c=}")
minus_ker((1,), (5,), (a, b, c, 5))
print(f"{c=}")

