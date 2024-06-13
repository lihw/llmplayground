/** 
* The misc stuff
*
* Author: lihw81@gmail.com
*/

#include <common/m_misc.h>

#include <spdlog/spdlog.h>
#include <fmt/core.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#endif

#undef min
#undef max

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


File::File(const char * fname, const char * mode) {
    name = fname;

#ifdef _WIN32
    fopen_s(&fp, fname, mode);
#else
    fp = fopen(fname, mode);
#endif

    if (fp == NULL) {
        char str[256];
        throw std::runtime_error(fmt::format("failed to open {}: {}", fname, strerror_s(str, sizeof(str), errno)));
    }
    seek(0, SEEK_END);
    size = tell();
    seek(0, SEEK_SET);
}

File::~File() {
    if (fp) {
        std::fclose(fp);
    }
}


size_t File::tell() const {
#ifdef _WIN32
    __int64 ret = _ftelli64(fp);
#else
    long ret = std::ftell(fp);
#endif
    assert(ret != -1); // this really shouldn't fail
    return (size_t) ret;
}

void File::seek(size_t offset, int whence) const {
#ifdef _WIN32
    int ret = _fseeki64(fp, (__int64) offset, whence);
#else
    int ret = std::fseek(fp, (long) offset, whence);
#endif
    assert(ret == 0); // same
}

void File::readRaw(void * ptr, size_t len) const {
    if (len == 0) {
        return;
    }
    errno = 0;
    std::size_t ret = std::fread(ptr, len, 1, fp);
    if (ferror(fp)) {
        char str[256];
        throw std::runtime_error(fmt::format("read error: {}", strerror_s(str, sizeof(str), errno)));
    }
    if (ret != 1) {
        throw std::runtime_error("unexpectedly reached end of file");
    }
}

uint32_t File::readU32() const {
    uint32_t ret;
    readRaw(&ret, sizeof(ret));
    return ret;
}

void File::writeRaw(const void * ptr, size_t len) const {
    if (len == 0) {
        return;
    }
    errno = 0;
    size_t ret = std::fwrite(ptr, len, 1, fp);
    if (ret != 1) {
        char str[256];
        throw std::runtime_error(fmt::format("write error: {}", strerror_s(str, sizeof(str), errno)));
    }
}

void File::writeU32(uint32_t val) const {
    writeRaw(&val, sizeof(val));
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

static std::string getWinErrString(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}

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
            spdlog::warn("warning: failed to VirtualLock %zu-byte buffer (after previously locking {} bytes): {}\n",
                len,
                size,
                getWinErrString(GetLastError()));
            return false;
        }

        // It failed but this was only the first try; increase the working
        // set size and try again.
        SIZE_T min_ws_size, max_ws_size;
        if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
            spdlog::warn(
                "GetProcessWorkingSetSize failed: {}", getWinErrString(GetLastError()));
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
            spdlog::warn(
                "SetProcessWorkingSetSize failed: {}", getWinErrString(GetLastError()));
            return false;
        }
    }
}

void MemoryLock::rawUnlock(void *ptr, size_t len)
{
    if (!VirtualUnlock(ptr, len)) {
        spdlog::warn("failed to VirtualUnlock buffer: {}", getWinErrString(GetLastError()));
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
    assert(addr == nullptr && len == 0);
}
#endif


#ifdef _POSIX_MAPPED_FILES
Mmap::Mmap(const std::string& file, size_t prefetch, bool numa) 
{
    size = file->size;

    int fd = fileno(file->fp);

    int flags = MAP_SHARED;
    // prefetch/readahead impairs performance on NUMA systems
    if (numa) { prefetch = 0; }
#ifdef __linux__
    // advise the kernel to read the file sequentially (increases readahead)
    if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
        LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                strerror(errno));
    }
    if (prefetch) { flags |= MAP_POPULATE; }
#endif
    addr = mmap(NULL, size, PROT_READ, flags, fd, 0);
    if (addr == MAP_FAILED) { // NOLINT
        throw std::runtime_error(fmt::format("mmap failed: {}", strerror(errno)));
    }

    if (prefetch > 0) {
        // advise the kernel to preload the mapped memory
        if (posix_madvise(addr, std::min(size, prefetch), POSIX_MADV_WILLNEED)) {
            spdlog::warn("posix_madvise(.., POSIX_MADV_WILLNEED) failed: {}",
                    strerror(errno));
        }
    }
    if (numa) {
        // advise the kernel not to use readahead
        // (because the next page might not belong on the same node)
        if (posix_madvise(addr, size, POSIX_MADV_RANDOM)) {
            spdlog::warn("posix_madvise(.., POSIX_MADV_RANDOM) failed: {}",
                    strerror(errno));
        }
    }

    // initialize list of mapped_fragments
    mapped_fragments.emplace_back(0, size);
}

Mmap::~Mmap() {
    for (const auto& frag : mappedFragments) {
        if (munmap((char*)addr + frag.first, frag.second - frag.first)) {
            spdlog::warn("munmap failed: {}", strerror(errno));
        }
    }
}

// partially unmap the file in the range [first, last)
void Mmap::unmapRange(size_t first, size_t last) {
    // note: this function must not be called multiple times with overlapping ranges
    // otherwise, there is a risk of invalidating addresses that have been repurposed for other mappings
    int page_size = sysconf(_SC_PAGESIZE);

    auto alignRange = [](size_t* first, size_t* last, size_t page_size) {
        // align first to the next page
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        // align last to the previous page
        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    };

    alignRange(&first, &last, page_size);
    size_t len = last - first;

    if (len == 0) {
        return;
    }

    GGML_ASSERT(first % page_size == 0);
    GGML_ASSERT(last % page_size == 0);
    GGML_ASSERT(last > first);

    void* next_page_start = (uint8_t*)addr + first;

    // unmap the range
    if (munmap(next_page_start, len)) {
        LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
    }

    // update the list of mapped fragments to avoid unmapping the same range again in the destructor
    std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
    for (const auto& frag : mapped_fragments) {
        if (frag.first < first && frag.second > last) {
            // the range is in the middle of the fragment, split it
            new_mapped_fragments.emplace_back(frag.first, first);
            new_mapped_fragments.emplace_back(last, frag.second);
        }
        else if (frag.first < first && frag.second > first) {
            // the range starts in the middle of the fragment
            new_mapped_fragments.emplace_back(frag.first, first);
        }
        else if (frag.first < last && frag.second > last) {
            // the range ends in the middle of the fragment
            new_mapped_fragments.emplace_back(last, frag.second);
        }
        else if (frag.first >= first && frag.second <= last) {
            // the range covers the entire fragment
        }
        else {
            // the range is outside the fragment
            new_mapped_fragments.push_back(frag);
        }
    }
    mapped_fragments = std::move(new_mapped_fragments);
}

#elif defined(_WIN32)

Mmap::Mmap(File* file, size_t prefetch, bool numa) {

    assert(numa == false);

    size = file->size;

    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (hMapping == NULL) {
        DWORD error = GetLastError();
        throw std::runtime_error(fmt::format("CreateFileMappingA failed: {}", getWinErrString(error)));
    }

    addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    DWORD error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
        throw std::runtime_error(fmt::format("MapViewOfFile failed: {}", getWinErrString(error)));
    }

    if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
        // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
        BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

        // may fail on pre-Windows 8 systems
        pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)> (GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

        if (pPrefetchVirtualMemory) {
            // advise the kernel to preload the mapped memory
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = addr;
            range.NumberOfBytes = (SIZE_T)std::min(size, prefetch);
            if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                spdlog::warn("PrefetchVirtualMemory failed: {}",
                    getWinErrString(GetLastError()));
            }
        }
#else
        throw std::runtime_error("PrefetchVirtualMemory unavailable");
#endif
    }
}

void Mmap::unmapRange(size_t first, size_t last) {
    assert(first == 0 && last == 0);
}

Mmap::~Mmap() {
    if (!UnmapViewOfFile(addr)) {
        spdlog::warn("UnmapViewOfFile failed: {}",
            getWinErrString(GetLastError()));
    }
}
#else

Mmap::Mmap(struct llama_file* file, size_t prefetch, bool numa) {
    GGML_UNUSED(file);
    GGML_UNUSED(prefetch);
    GGML_UNUSED(numa);

    throw std::runtime_error("mmap not supported");
}

void Mmap::unmapRange(size_t first, size_t last) {
    GGML_UNUSED(first);
    GGML_UNUSED(last);

    throw std::runtime_error("mmap not supported");
}
#endif

M_END_NAMESPACE

