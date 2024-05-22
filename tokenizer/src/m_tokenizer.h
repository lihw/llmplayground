/** 
* The abstract interface of tokenizer class
*
* Author: lihw81@gmail.com
*/

#ifndef M_TOKENIZER_H
#define M_TOKENIZER_H

#include "common/defs.h"

class Tokenizer {
    NO_COPY_CLASS(Tokenizer)
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    const Tokenizer& operator=(const Tokenizer&) = delete;
    const Tokenizer& operator=(Tokenizer&&) = delete;
};

#endif // !M_TOKENIZER_H