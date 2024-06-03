/** 
* The misc stuff
*
* Author: lihw81@gmail.com
*/

#include <common/m_misc.h>

M_BEGIN_NAMESPACE

void replaceAll(std::string& s, const std::string& search, const std::string& replace) 
{
    std::string result;
    for (size_t pos = 0; ; pos += search.length()) {
        auto new_pos = s.find(search, pos);
        if (new_pos == std::string::npos) {
            result += s.substr(pos, s.size() - pos);
            break;
        }
        result += s.substr(pos, new_pos - pos) + replace;
        pos = new_pos;
    }
    s = std::move(result);
}
    
size_t utf8Len(char src) noexcept 
{
        const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
        uint8_t highbits = static_cast<uint8_t>(src) >> 4;
        return lookup[highbits];
    }

void escapeWhitespace(std::string & text) 
{
    replaceAll(text, " ", "\xe2\x96\x81");
}

void unescapeWhitespace(std::string & word) 
{
    replaceAll(word, "\xe2\x96\x81", " ");
}

M_END_NAMESPACE

