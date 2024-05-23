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
    TYPE_UNDEFINED    = 0,
    TYPE_NORMAL       = 1,
    TYPE_UNKNOWN      = 2,
    TYPE_CONTROL      = 3,
    TYPE_USER_DEFINED = 4,
    TYPE_UNUSED       = 5,
    TYPE_BYTE         = 6,
};

struct TokenData {
    Token text;
    float score;
    TokenType type;
};


M_END_NAMESPACE

#endif //! M_DEFS_H
