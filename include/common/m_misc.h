/**
* Misc functions
* Author: lihw81@gmail.com
*/


#ifndef M_MISC_H
#define M_MISC_H

#include "m_defs.h"

M_BEGIN_NAMESPACE

extern void replaceAll(std::string& s, const std::string& search, const std::string& replace);

extern size_t utf8Len(char src) noexcept;

extern void escapeWhitespace(std::string & text);
extern void unescapeWhitespace(std::string & word);

M_END_NAMESPACE

#endif //! M_MISC_H
