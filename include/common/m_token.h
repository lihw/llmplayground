/**
* The token
* Author: lihw81@gmail.com
*/


#ifndef M_TOKEN_H
#define M_TOKEN_H

#include "m_defs.h"

#include <string>

M_BEGIN_NAMESPACE

using Token = std::string;
using TokenId = int32_t;

enum class TokenType {
    UNDEFINED    = 0,
    NORMAL       = 1,
    UNKNOWN      = 2,
    CONTROL      = 3,
    USER_DEFINED = 4,
    UNUSED       = 5,
    BYTE         = 6,
};

struct TokenData {
    Token text;
    float score;
    TokenType type;
};


M_END_NAMESPACE

#endif //! M_DEFS_H
