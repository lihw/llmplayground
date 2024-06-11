/** 
* The misc stuff
*
* Author: lihw81@gmail.com
*/

#include <common/m_misc.h>

#include <spdlog/spdlog.h>

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

#ifdef _POSIX_MEMLOCK_RANGE
size_t MemoryLock::lockGranularity() 
{
    return (size_t) sysconf(_SC_PAGESIZE);
}

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                                                            \
    "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
    "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

bool MemoryLock::rawLock(const void *addr, size_t size) const
{
    if (!mlock(addr, size)) {
        return true;
    }

    char *errmsg = std::strerror(errno);
    bool suggest = (errno == ENOMEM);

    // Check if the resource limit is fine after all
    struct rlimit lockLimit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lockLimit)) {
        suggest = false;
    }
    if (suggest && (lockLimit.rlimMax > lockLimit.rlimCur + size)) {
        suggest = false;
    }

    spdlog::warn("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
        size,
        this->size,
        errmsg,
        suggest ? MLOCK_SUGGESTION : "");
    return false;
}

#undef MLOCK_SUGGESTION

void MemoryLock::rawUnlock(void *addr, size_t size)
{
    if (munlock(addr, size)) {
        spdlog::warn("warning: failed to munlock buffer: %s\n", std::strerror(errno));
    }
}

#elif defined(_WIN32)

size_t MemoryLock::lockGranularity() 
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
}

bool MemoryLock::rawLock(void * ptr, size_t len) const
{
    for (int tries = 1;; tries++) {
        if (VirtualLock(ptr, len)) {
            return true;
        }
        if (tries == 2) {
            LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                len,
                size,
                llama_format_win_err(GetLastError()).c_str());
            return false;
        }

        // It failed but this was only the first try; increase the working
        // set size and try again.
        SIZE_T min_ws_size, max_ws_size;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
            LLAMA_LOG_WARN(
                "warning: GetProcessWorkingSetSize failed: %s\n", llama_format_win_err(GetLastError()).c_str());
            return false;
        }
        // Per MSDN: "The maximum number of pages that a process can lock
        // is equal to the number of pages in its minimum working set minus
        // a small overhead."
        // Hopefully a megabyte is enough overhead:
        size_t increment = len + 1048576;
        // The minimum must be <= the maximum, so we need to increase both:
        min_ws_size += increment;
        max_ws_size += increment;
        if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
            LLAMA_LOG_WARN(
                "warning: SetProcessWorkingSetSize failed: %s\n", llama_format_win_err(GetLastError()).c_str());
            return false;
        }
    }
}

void MemoryLock::rawUnlock(void *ptr, size_t len)
{
    if (!VirtualUnlock(ptr, len)) {
        LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n", llama_format_win_err(GetLastError()).c_str());
    }
}

#else

size_t MemoryLock::lockGranularity() { return (size_t)65536; }

bool MemoryLock::rawLock(const void *addr, size_t len) const
{
    spdlog::warn("warning: mlock not supported on this system");
    return false;
}

void MemoryLock::rawUnlock(const void * addr, size_t len) 
{
}
#endif

M_END_NAMESPACE

