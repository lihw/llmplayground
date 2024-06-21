/**
* Misc functions
* Author: lihw81@gmail.com
*/


#ifndef M_MISC_H
#define M_MISC_H

#include "m_defs.h"

#include <vector>
#include <memory>

#include <cassert>

M_BEGIN_NAMESPACE

#define M_NOTUSE(x) (void)(x)

extern void replaceAll(std::string& s, const std::string& search, const std::string& replace);

extern size_t utf8Len(char src) noexcept;

extern void escapeWhitespace(std::string & text);
extern void unescapeWhitespace(std::string & word);

struct File {
    std::string name; //! The file path
    FILE * fp; //! use FILE * so we don't have to re-open the file to mmap
    size_t size;
    
    File(const char * fname, const char * mode);
    ~File();
    
    size_t tell() const;
    void seek(size_t offset, int whence) const;
    void readRaw(void * ptr, size_t len) const;
    uint32_t readU32() const;
    void writeRaw(const void * ptr, size_t len) const;
    void writeU32(uint32_t val) const;
};
using Files = std::vector<std::unique_ptr<File>>;

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

    bool rawLock(void * addr, size_t len) const;

    static void rawUnlock(void* addr, size_t len);

#if defined _POSIX_MEMLOCK_RANGE || defined _WIN32
    static constexpr bool SUPPORTED = true;
#else
    static constexpr bool SUPPORTED = false;

#endif
};
using MemoryLocks = std::vector<std::unique_ptr<MemoryLock>>;

struct Mmap {
    void* addr;
    size_t size;

    M_NO_COPY_CONSTRUCTOR(Mmap);
    M_NO_MOVE_CONSTRUCTOR(Mmap);

    Mmap(File* file, size_t prefetch = (size_t)-1 /* -1 = max value */, bool numa = false);

    ~Mmap();

    // partially unmap the file in the range [first, last)
    void unmapRange(size_t first, size_t last);
    

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;
    // list of mapped fragments (first_offset, last_offset)
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

#else
    static constexpr bool SUPPORTED = false;
#endif
};

using Mmaps = std::vector<std::unique_ptr<Mmap>>;

M_END_NAMESPACE

#endif //! M_MISC_H
