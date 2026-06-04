#include "OPERATOR/Pool.hpp"

#if CUDA_CPU == 1
static void helperSetTensor4d(cudnnTensorDescriptor_t desc, VIEW::Math *math)
{
  VIEW::Shape& layout = math->getLayout();
  cudnnSetTensor4dDescriptor(
    desc,
    CUDNN_TENSOR_NCHW,
    CUDNN_DATA_HALF,
    layout.getDim(0),
    layout.getDim(1),
    layout.getDim(2),
    layout.getDim(3));
}
#else
#endif

OPERATOR::Pool::Pool(HANDLER::Workspace& workspace)
{
  this->workspace = &workspace;
  this->windowH = 2;
  this->windowW = 2;
  this->padH = 0;
  this->padW = 0;
  this->strideH = 2;
  this->strideW = 2;
#if CUDA_CPU == 1
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreatePoolingDescriptor(&this->poolDesc);
#else
#endif
}

OPERATOR::Pool::Pool(HANDLER::Workspace& workspace, VIEW::Math& out, VIEW::Math& in)
{
  this->workspace = &workspace;
  this->out = &out;
  this->in = &in;
  this->windowH = 2;
  this->windowW = 2;
  this->padH = 0;
  this->padW = 0;
  this->strideH = 2;
  this->strideW = 2;
#if CUDA_CPU == 1
  cudnnCreateTensorDescriptor(&this->xDesc);
  cudnnCreateTensorDescriptor(&this->yDesc);
  cudnnCreatePoolingDescriptor(&this->poolDesc);
#else
#endif
  this->setDescriptor();
}

OPERATOR::Pool::~Pool()
{
#if CUDA_CPU == 1
  cudnnDestroyPoolingDescriptor(this->poolDesc);
  cudnnDestroyTensorDescriptor(this->yDesc);
  cudnnDestroyTensorDescriptor(this->xDesc);
#else
#endif
}

VIEW::Math& OPERATOR::Pool::getInput()
{
  return *this->in;
}

VIEW::Math& OPERATOR::Pool::getOutput()
{
  return *this->out;
}

VIEW::Math& OPERATOR::Pool::getGradInput()
{
  return *this->dIn;
}

VIEW::Math& OPERATOR::Pool::getGradOutput()
{
  return *this->dOut;
}

void OPERATOR::Pool::setOperand(VIEW::Math& out, VIEW::Math& in)
{
  this->out = &out;
  this->in = &in;
  this->setDescriptor();
}

void OPERATOR::Pool::setGradOperand(VIEW::Math& dIn, VIEW::Math& dOut)
{
  this->dIn = &dIn;
  this->dOut = &dOut;
}

void OPERATOR::Pool::setConfig(int windowH, int windowW, int padH, int padW, int strideH, int strideW)
{
  this->windowH = windowH;
  this->windowW = windowW;
  this->padH = padH;
  this->padW = padW;
  this->strideH = strideH;
  this->strideW = strideW;
  this->setDescriptor();
}

void OPERATOR::Pool::setDescriptor()
{
  if(this->in == NULL || this->out == NULL) return;

#if CUDA_CPU == 1
  helperSetTensor4d(this->xDesc, this->in);
  helperSetTensor4d(this->yDesc, this->out);
  cudnnSetPooling2dDescriptor(
    this->poolDesc,
    CUDNN_POOLING_MAX,
    CUDNN_PROPAGATE_NAN,
    this->windowH,
    this->windowW,
    this->padH,
    this->padW,
    this->strideH,
    this->strideW);
#else
#endif
}

void OPERATOR::Pool::maxForward()
{
#if CUDA_CPU == 1
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnPoolingForward(
    this->workspace->getCudnnHandle(),
    this->poolDesc,
    &alpha,
    this->xDesc,
    this->in->getGpuPtr(),
    &beta,
    this->yDesc,
    this->out->getGpuPtr());
#else
#endif
}

void OPERATOR::Pool::maxBackward()
{
#if CUDA_CPU == 1
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnPoolingBackward(
    this->workspace->getCudnnHandle(),
    this->poolDesc,
    &alpha,
    this->yDesc,
    this->out->getGpuPtr(),
    this->yDesc,
    this->dOut->getGpuPtr(),
    this->xDesc,
    this->in->getGpuPtr(),
    &beta,
    this->xDesc,
    this->dIn->getGpuPtr());
#else
#endif
}
