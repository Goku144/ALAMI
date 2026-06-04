#include "HANDLER/Workspace.hpp"

static const char *getWorkspaceErrorMessage(CORE::errWorkspace err)
{
  if(err == CORE::workspaceSuccess) return "Workspace success";

#if CUDA_CPU == 1
  if(err == CORE::workspaceErrCudnnCreate) return "Workspace failed to create cuDNN handle";
  if(err == CORE::workspaceErrCudnnDestroy) return "Workspace failed to destroy cuDNN handle";
  if(err == CORE::workspaceErrCublasLtCreate) return "Workspace failed to create cuBLASLt handle";
  if(err == CORE::workspaceErrCublasLtDestroy) return "Workspace failed to destroy cuBLASLt handle";
  if(err == CORE::workspaceErrCudnnSetStream) return "Workspace failed to attach stream to cuDNN handle";
  if(err == CORE::workspaceErrCudaStreamCreate) return "Workspace failed to create CUDA stream";
  if(err == CORE::workspaceErrCudaStreamDestroy) return "Workspace failed to destroy CUDA stream";
#else
  if(err == CORE::workspaceErrDnnlEngineCreate) return "Workspace failed to create oneDNN engine";
  if(err == CORE::workspaceErrDnnlEngineDestroy) return "Workspace failed to destroy oneDNN engine";
  if(err == CORE::workspaceErrDnnlStreamCreate) return "Workspace failed to create oneDNN stream";
  if(err == CORE::workspaceErrDnnlStreamDestroy) return "Workspace failed to destroy oneDNN stream";
#endif

  if(err == CORE::workspaceErrScratchAlloc) return "Workspace failed to allocate scratch memory";
  if(err == CORE::workspaceErrScratchFree) return "Workspace failed to free scratch memory";
  if(err == CORE::workspaceErrScratchOutOfBound) return "Workspace scratch memory is too small";
  if(err == CORE::workspaceErrNull) return "Workspace null pointer";

  return "Workspace unknown error";
}

HANDLER::Workspace::Workspace(HANDLER::IO& io, HANDLER::File& file, size_t scratchBytes)
{
  this->io = &io;
  this->file = &file;
  file.setIO(io);

#if CUDA_CPU == 1
  if(cudaStreamCreate(&this->stream) != cudaSuccess)
  {
    this->err = CORE::workspaceErrCudaStreamCreate;
    return;
  }

  if(cudnnCreate(&this->cudnnHandle) != CUDNN_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCudnnCreate;
    return;
  }

  if(cublasLtCreate(&this->cublasLtHandle) != CUBLAS_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCublasLtCreate;
    return;
  }

  if(cudnnSetStream(this->cudnnHandle, this->stream) != CUDNN_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCudnnSetStream;
    return;
  }

  this->scratchBytes = scratchBytes;
  if(scratchBytes > 0 && cudaMalloc(&this->scratchPtr, scratchBytes) != cudaSuccess)
  {
    this->scratchBytes = 0;
    this->err = CORE::workspaceErrScratchAlloc;
  }
#else
  if(dnnl_engine_create(&this->engine, dnnl_cpu, 0) != dnnl_success)
  {
    this->err = CORE::workspaceErrDnnlEngineCreate;
    return;
  }

  if(dnnl_stream_create(
        &this->stream,
        this->engine,
        dnnl_stream_default_flags) != dnnl_success)
  {
    this->err = CORE::workspaceErrDnnlStreamCreate;
    return;
  }

  this->scratchBytes =
      CORE::ALIGNE(scratchBytes, CORE::ALIGNE_TO_256);

  if(this->scratchBytes > 0)
  {
    this->scratchPtr =
        aligned_alloc(CORE::ALIGNE_TO_256, this->scratchBytes);

    if(this->scratchPtr == NULL)
    {
      this->scratchBytes = 0;
      this->err = CORE::workspaceErrScratchAlloc;
    }
  }
#endif
}

HANDLER::Workspace::~Workspace()
{
#if CUDA_CPU == 1
  if(this->scratchPtr != NULL && cudaFree(this->scratchPtr) != cudaSuccess)
  {
    this->err = CORE::workspaceErrScratchFree;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->cublasLtHandle != NULL && cublasLtDestroy(this->cublasLtHandle) != CUBLAS_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCublasLtDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->cudnnHandle != NULL && cudnnDestroy(this->cudnnHandle) != CUDNN_STATUS_SUCCESS)
  {
    this->err = CORE::workspaceErrCudnnDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->stream != NULL && cudaStreamDestroy(this->stream) != cudaSuccess)
  {
    this->err = CORE::workspaceErrCudaStreamDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }
#else
  if(this->stream != NULL) dnnl_stream_wait(this->stream);

  if(this->stream != NULL && dnnl_stream_destroy(this->stream) != dnnl_success)
  {
    this->err = CORE::workspaceErrDnnlStreamDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  if(this->engine != NULL && dnnl_engine_destroy(this->engine) != dnnl_success)
  {
    this->err = CORE::workspaceErrDnnlEngineDestroy;
    CORE::logWarn(__FILE__, __LINE__, "%s", getWorkspaceErrorMessage(this->err));
  }

  free(this->scratchPtr);
#endif
}

HANDLER::IO& HANDLER::Workspace::getIO()
{
  return *this->io;
}

HANDLER::File& HANDLER::Workspace::getFile()
{
  return *this->file;
}

#if CUDA_CPU == 1
cudnnHandle_t HANDLER::Workspace::getCudnnHandle()
{
  return this->cudnnHandle;
}

cublasLtHandle_t HANDLER::Workspace::getCublasLtHandle()
{
  return this->cublasLtHandle;
}

cudaStream_t HANDLER::Workspace::getStream()
{
  return this->stream;
}
#else
dnnl_stream_t HANDLER::Workspace::getStream()
{
  return this->stream;
}

dnnl_engine_t HANDLER::Workspace::getEngine()
{
  return this->engine;
}
#endif

void *HANDLER::Workspace::getScratch(size_t requiredBytes)
{
  if(requiredBytes == 0) return NULL;

  if(this->scratchPtr == NULL)
  {
    this->err = CORE::workspaceErrNull;
    return NULL;
  }

  if(requiredBytes > this->scratchBytes)
  {
    this->err = CORE::workspaceErrScratchOutOfBound;
    return NULL;
  }

  return this->scratchPtr;
}

size_t HANDLER::Workspace::getScratchBytes() const
{
  return this->scratchBytes;
}

CORE::errWorkspace HANDLER::Workspace::getErr()
{
  CORE::errWorkspace err = this->err;
  this->err = CORE::workspaceSuccess;
  return err;
}

CORE::errWorkspace HANDLER::Workspace::peekErr() const
{
  return this->err;
}

void HANDLER::Workspace::clearErr()
{
  this->err = CORE::workspaceSuccess;
}

void HANDLER::Workspace::info(CORE::State level, const char *file, int line) const
{
  if(this->err == CORE::workspaceSuccess) return;

  if(level == CORE::FATAL)
  {
    CORE::logFatal(file, line, "%s", getWorkspaceErrorMessage(this->err));
    return;
  }

  CORE::logWarn(file, line, "%s", getWorkspaceErrorMessage(this->err));
}
