#include "OPERATOR/MatrixMulBias.hpp"

#if CUDA_CPU == 1
#include <cuda_fp16.h>
#include <cuda/cmath>
#else
#include <dnnl.hpp>
#endif

static int matrixRows(VIEW::Math *math)
{
  VIEW::Shape& layout = math->getLayout();
  return layout.getRank() > 1 ? layout.getDim(0) : 1;
}

static int matrixCols(VIEW::Math *math)
{
  VIEW::Shape& layout = math->getLayout();
  return layout.getRank() > 1 ? layout.getDim(1) : layout.getDim(0);
}

#if CUDA_CPU == 1
static void clearMatmulLtDescriptor(
  cublasLtMatmulDesc_t& operationDesc,
  cublasLtMatrixLayout_t& aDesc,
  cublasLtMatrixLayout_t& bDesc,
  cublasLtMatrixLayout_t& cDesc)
{
  if(cDesc != NULL) cublasLtMatrixLayoutDestroy(cDesc);
  if(bDesc != NULL) cublasLtMatrixLayoutDestroy(bDesc);
  if(aDesc != NULL) cublasLtMatrixLayoutDestroy(aDesc);
  if(operationDesc != NULL) cublasLtMatmulDescDestroy(operationDesc);

  operationDesc = NULL;
  aDesc = NULL;
  bDesc = NULL;
  cDesc = NULL;
}

static void setMatmulLtDescriptor(
  HANDLER::Workspace *workspace,
  cublasLtMatmulPreference_t preference,
  cublasLtMatmulDesc_t& operationDesc,
  cublasLtMatrixLayout_t& aDesc,
  cublasLtMatrixLayout_t& bDesc,
  cublasLtMatrixLayout_t& cDesc,
  cublasLtMatmulHeuristicResult_t& heuristic,
  int aRows,
  int aCols,
  int bRows,
  int bCols,
  int cRows,
  int cCols,
  cublasOperation_t transA,
  cublasOperation_t transB,
  void *biasPtr)
{
  clearMatmulLtDescriptor(operationDesc, aDesc, bDesc, cDesc);
  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

  if(biasPtr != NULL)
  {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr));
  }

  cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, aRows, aCols, aRows);
  cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, bRows, bCols, bRows);
  cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16F, cRows, cCols, cRows);

  int returnedResults = 0;
  cublasLtMatmulAlgoGetHeuristic(
    workspace->getCublasLtHandle(),
    operationDesc,
    aDesc,
    bDesc,
    cDesc,
    cDesc,
    preference,
    1,
    &heuristic,
    &returnedResults);

  if(returnedResults == 0) clearMatmulLtDescriptor(operationDesc, aDesc, bDesc, cDesc);
}

static void runMatmulLt(
  HANDLER::Workspace *workspace,
  cublasLtMatmulDesc_t operationDesc,
  cublasLtMatrixLayout_t aDesc,
  cublasLtMatrixLayout_t bDesc,
  cublasLtMatrixLayout_t cDesc,
  cublasLtMatmulHeuristicResult_t& heuristic,
  VIEW::Math *out,
  VIEW::Math *a,
  VIEW::Math *b)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;

  if(operationDesc != NULL)
  {
    cublasLtMatmul(
      workspace->getCublasLtHandle(),
      operationDesc,
      &alpha,
      a->getGpuPtr(),
      aDesc,
      b->getGpuPtr(),
      bDesc,
      &beta,
      out->getGpuPtr(),
      cDesc,
      out->getGpuPtr(),
      cDesc,
      &heuristic.algo,
      workspace->getScratch(heuristic.workspaceSize),
      heuristic.workspaceSize,
      workspace->getStream());
  }
}
#else
#endif

void OPERATOR::MatrixMulBias::clearDescriptor()
{
#if CUDA_CPU == 1
  clearMatmulLtDescriptor(this->forwardOperationDesc, this->forwardADesc, this->forwardBDesc, this->forwardCDesc);
  clearMatmulLtDescriptor(this->gradInputOperationDesc, this->gradInputADesc, this->gradInputBDesc, this->gradInputCDesc);
  clearMatmulLtDescriptor(this->gradWeightOperationDesc, this->gradWeightADesc, this->gradWeightBDesc, this->gradWeightCDesc);
#else
#endif
}

void OPERATOR::MatrixMulBias::setDescriptor()
{
#if CUDA_CPU == 1
  this->clearDescriptor();

  if(this->in == NULL || this->weight == NULL || this->bias == NULL || this->out == NULL) return;

  int batch = matrixRows(this->in);
  int inFeatures = matrixCols(this->in);
  int outFeatures = matrixCols(this->weight);

  setMatmulLtDescriptor(
    this->workspace, this->preference,
    this->forwardOperationDesc, this->forwardADesc, this->forwardBDesc, this->forwardCDesc,
    this->forwardHeuristic,
    outFeatures, inFeatures, inFeatures, batch, outFeatures, batch,
    CUBLAS_OP_N, CUBLAS_OP_N, this->bias->getGpuPtr());

  if(this->dIn == NULL || this->dWeight == NULL || this->dOut == NULL) return;

  setMatmulLtDescriptor(
    this->workspace, this->preference,
    this->gradInputOperationDesc, this->gradInputADesc, this->gradInputBDesc, this->gradInputCDesc,
    this->gradInputHeuristic,
    outFeatures, inFeatures, outFeatures, batch, inFeatures, batch,
    CUBLAS_OP_T, CUBLAS_OP_N, NULL);

  setMatmulLtDescriptor(
    this->workspace, this->preference,
    this->gradWeightOperationDesc, this->gradWeightADesc, this->gradWeightBDesc, this->gradWeightCDesc,
    this->gradWeightHeuristic,
    outFeatures, batch, inFeatures, batch, outFeatures, inFeatures,
    CUBLAS_OP_N, CUBLAS_OP_T, NULL);
#else
#endif
}

#if CUDA_CPU == 1
static __device__ __forceinline__ unsigned int helperKernelHalf2ToU32(__half2 x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.h = x;
  return v.u;
}

static __device__ __forceinline__ __half2 helperKernelU32ToHalf2(unsigned int x)
{
  union
  {
    unsigned int u;
    __half2 h;
  } v;
  v.u = x;
  return v.h;
}

static __device__ __forceinline__ void helperKernelAddHalf2(float& lo, float& hi, unsigned int x)
{
  __half2 h = helperKernelU32ToHalf2(x);
  lo += __half2float(__low2half(h));
  hi += __half2float(__high2half(h));
}

static __device__ __forceinline__ unsigned int helperKernelFloat2ToU32(float lo, float hi)
{
  return helperKernelHalf2ToU32(__halves2half2(__float2half(lo), __float2half(hi)));
}

static __global__ void matrixMulBiasGradBiasKernel(uint4 * __restrict__ DBIAS, const uint4 * __restrict__ DOUT, int batch, int features8)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= features8) return;

  float x0 = 0.0f;
  float x1 = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
  float z0 = 0.0f;
  float z1 = 0.0f;
  float w0 = 0.0f;
  float w1 = 0.0f;

  for(int row = 0; row < batch; row++)
  {
    uint4 dOut = DOUT[row * features8 + index];
    helperKernelAddHalf2(x0, x1, dOut.x);
    helperKernelAddHalf2(y0, y1, dOut.y);
    helperKernelAddHalf2(z0, z1, dOut.z);
    helperKernelAddHalf2(w0, w1, dOut.w);
  }

  uint4 dBias;
  dBias.x = helperKernelFloat2ToU32(x0, x1);
  dBias.y = helperKernelFloat2ToU32(y0, y1);
  dBias.z = helperKernelFloat2ToU32(z0, z1);
  dBias.w = helperKernelFloat2ToU32(w0, w1);
  DBIAS[index] = dBias;
}

static __global__ void matrixMulBiasGradBiasTailKernel(__half * __restrict__ DBIAS, const __half * __restrict__ DOUT, int batch, int features, int tailStart, int tailN)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if(index >= tailN) return;

  int col = tailStart + index;
  float acc = 0.0f;
  for(int row = 0; row < batch; row++)
  {
    acc += __half2float(DOUT[row * features + col]);
  }
  DBIAS[col] = __float2half_rn(acc);
}
#else
#endif

OPERATOR::MatrixMulBias::MatrixMulBias(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
#if CUDA_CPU == 1
  cublasLtMatmulPreferenceCreate(&this->preference);
  size_t workspaceBytes = workspace.getScratchBytes();
  cublasLtMatmulPreferenceSetAttribute(this->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));
#else
#endif
}

OPERATOR::MatrixMulBias::MatrixMulBias(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
  this->weight = &weight;
  this->bias = &bias;
#if CUDA_CPU == 1
  cublasLtMatmulPreferenceCreate(&this->preference);
  size_t workspaceBytes = workspace.getScratchBytes();
  cublasLtMatmulPreferenceSetAttribute(this->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));
#else
#endif
  this->setDescriptor();
}

OPERATOR::MatrixMulBias::~MatrixMulBias()
{
  this->clearDescriptor();
#if CUDA_CPU == 1
  if(this->preference != NULL) cublasLtMatmulPreferenceDestroy(this->preference);
#else
#endif
}

VIEW::Math& OPERATOR::MatrixMulBias::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::MatrixMulBias::getWeight()
{
  return *this->weight;
}

VIEW::Math& OPERATOR::MatrixMulBias::getBias()
{
  return *this->bias;
}

VIEW::Math& OPERATOR::MatrixMulBias::getOutput()
{
  return *this->out;
}

VIEW::Math& OPERATOR::MatrixMulBias::getGradInput()
{
  return *this->dIn;
}

VIEW::Math& OPERATOR::MatrixMulBias::getGradWeight()
{
  return *this->dWeight;
}

VIEW::Math& OPERATOR::MatrixMulBias::getGradBias()
{
  return *this->dBias;
}

VIEW::Math& OPERATOR::MatrixMulBias::getGradOutput()
{
  return *this->dOut;
}

void OPERATOR::MatrixMulBias::setOperand(VIEW::Math& out, VIEW::Math& in, VIEW::Math& weight, VIEW::Math& bias)
{
  this->out = &out;
  this->in = &in;
  this->weight = &weight;
  this->bias = &bias;
  this->setDescriptor();
}

void OPERATOR::MatrixMulBias::setGradOperand(VIEW::Math& dIn, VIEW::Math& dWeight, VIEW::Math& dBias, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dWeight = &dWeight;
  this->dBias = &dBias;
  this->dOut = &dOut;
  this->setDescriptor();
}

void OPERATOR::MatrixMulBias::forward()
{
#if CUDA_CPU == 1
  runMatmulLt(
    this->workspace,
    this->forwardOperationDesc,
    this->forwardADesc,
    this->forwardBDesc,
    this->forwardCDesc,
    this->forwardHeuristic,
    this->out,
    this->weight,
    this->in);
#else
#endif
}

void OPERATOR::MatrixMulBias::backward()
{
#if CUDA_CPU == 1
  int batch = matrixRows(this->in);

  runMatmulLt(
    this->workspace,
    this->gradInputOperationDesc,
    this->gradInputADesc,
    this->gradInputBDesc,
    this->gradInputCDesc,
    this->gradInputHeuristic,
    this->dIn,
    this->weight,
    this->dOut);

  runMatmulLt(
    this->workspace,
    this->gradWeightOperationDesc,
    this->gradWeightADesc,
    this->gradWeightBDesc,
    this->gradWeightCDesc,
    this->gradWeightHeuristic,
    this->dWeight,
    this->dOut,
    this->in);

  int features = matrixCols(this->dOut);
  int features8 = features / 8;
  int threads = 256;

  if(features8 > 0)
  {
    int blocks = cuda::ceil_div(features8, threads);
    matrixMulBiasGradBiasKernel<<<blocks, threads, 0, this->workspace->getStream()>>>(
      (uint4 *)this->dBias->getGpuPtr(),
      (const uint4 *)this->dOut->getGpuPtr(),
      batch,
      features8);
  }

  int tailStart = features8 * 8;
  if(tailStart < features)
  {
    int tailN = features - tailStart;
    int blocks = cuda::ceil_div(tailN, threads);
    matrixMulBiasGradBiasTailKernel<<<blocks, threads, 0, this->workspace->getStream()>>>(
      (__half *)this->dBias->getGpuPtr(),
      (const __half *)this->dOut->getGpuPtr(),
      batch,
      features,
      tailStart,
      tailN);
  }
#else
#endif
}
