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

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct MemoryLock {
    void * addr = NULL;
    size_t size = 0;

    bool failedAlready = false;

    MemoryLock() {}
    MemoryLock(const MemoryLock &) = delete;

    ~MemoryLock() {
        if (size) {
            rawUnlock(addr, size);
        }
    }

    void initialize(void * ptr) {
        assert(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    void growTo(size_t targetSize) {
        assert(addr);
        if (failedAlready) {
            return;
        }
        size_t granularity = lockGranularity();
        targetSize = (targetSize + granularity - 1) & ~(granularity - 1);
        if (targetSize > size) {
            if (rawLock((uint8_t *) addr + size, targetSize - size)) {
                size = targetSize;
            } else {
                failedAlready = true;
            }
        }
    }
    
    static size_t lockGranularity();

    bool rawLock(const void * addr, size_t len) const;

    static void rawUnlock(const void * addr, size_t len) {}

#if defined _POSIX_MEMLOCK_RANGE || defined _WIN32
    static constexpr bool SUPPORTED = true;
#else
    static constexpr bool SUPPORTED = false;

#endif
};
using MemoryLocks = std::vector<std::unique_ptr<MemoryLock>>;

M_END_NAMESPACE

#endif //! M_MISC_H
