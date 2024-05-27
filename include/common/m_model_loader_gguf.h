/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#ifndef M_MODEL_LOADER_GGUF_H
#define M_MODEL_LOADER_GGUF_H

#include "m_model_loader.h"

M_BEGIN_NAMESPACE

class ModelLoaderGguf : public MeshLoader final {
    M_NO_COPY_CONSTRUCTOR(ModelLoaderGguf)
    M_NO_MOVE_CONSTRUCTOR(ModelLoaderGguf)

public:
    explicit ModelLoaderGguf();

    ~ModelLoaderGguf();

    virtual bool load(const std::string& file, bool useMmap) noexcept final;

private:
    gguf_context* mMeta = nullptr;
};

M_END_NAMESPACE

#endif // !M_MODEL_LOADER_GGUF_H

