#include <benchmark/benchmark.h>
#include <random>
#include <memory>
#include <immintrin.h>

static constexpr int N = 1024;

alignas(64) float A[N * N];
alignas(64) float B[N * N];
alignas(64) float C[N * N];

static constexpr float scale = 100.0f / static_cast<float>(RAND_MAX);

void setup(const benchmark::State &)
{
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      A[j * N + i] = rand() * scale;
      A[j * N + i] = rand() * scale;
    }
  }
}

void mult_naive()
{
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      float acc = 0;
      for (int k = 0; k < N; ++k)
      {
        acc += A[j * N + k] * B[k * N + i];
      }
      C[j * N + i] = acc;
    }
  }
}

void bm_naive(benchmark::State &s)
{
  for (auto _ : s)
  {
    mult_naive();
  }
}

void mult_transpose()
{
  auto Bt = std::make_unique_for_overwrite<float[]>(N * N);
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      Bt[i * N + j] = B[j * N + i];
    }
  }
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      float acc = 0;
      for (int k = 0; k < N; ++k)
      {
        acc += A[j * N + k] * Bt[i * N + k];
      }
      C[j * N + i] = acc;
    }
  }
}

void bm_transpose(benchmark::State &s)
{
  for (auto _ : s)
  {
    mult_transpose();
  }
}

void mult_avx()
{
  float zero = 0.0f;
  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; i += 8)
    {
      __m256 acc = _mm256_broadcast_ss(&zero);
      for (int k = 0; k < N; ++k)
      {
        __m256 as = _mm256_broadcast_ss(&A[j * N + k]);
        auto bs = _mm256_load_ps(&B[k * N + i]);
        auto ms = _mm256_mul_ps(as, bs);
        acc = _mm256_add_ps(ms, acc);
      }
      _mm256_store_ps(&C[j * N + i], acc);
    }
  }
}

void bm_avx(benchmark::State &s)
{
  for (auto _ : s)
  {
    mult_avx();
  }
}

BENCHMARK(bm_naive)->Setup(setup);
BENCHMARK(bm_transpose)->Setup(setup);
BENCHMARK(bm_avx)->Setup(setup);

BENCHMARK_MAIN();
