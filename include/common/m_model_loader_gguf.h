/** 
* Load the GGUF models from files
*
* Author: lihw81@gmail.com
*/

#ifndef M_MODEL_LOADER_GGUF_H
#define M_MODEL_LOADER_GGUF_H

#include "m_model_loader.h"

M_BEGIN_NAMESPACE

class ModelLoaderGguf final : public ModelLoader {
    M_NO_COPY_CONSTRUCTOR(ModelLoaderGguf)
    M_NO_MOVE_CONSTRUCTOR(ModelLoaderGguf)

public:
    explicit ModelLoaderGguf();

    ~ModelLoaderGguf();

    virtual bool load(const std::string& file) noexcept final;

    virtual Model* build() noexcept final;

private:
    GgufVersion mVersion = GgufVersion::V1;
    
};

extern std::string ggufKvToStr(const struct gguf_context * ctx_gguf, int i); 

M_END_NAMESPACE

#endif // !M_MODEL_LOADER_GGUF_H

