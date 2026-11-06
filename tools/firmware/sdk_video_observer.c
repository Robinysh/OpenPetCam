/* Read-only, bounded observation at the T23 IMP encoder boundary.
 * This never changes SDK packs, stream counts, timestamps, flags or payloads.
 * ABI: pinned ingenic-headers T23 IMPEncoderPack / IMPEncoderStream.
 * Enable only for a maintenance capture with OPC_VIDEO_TRACE=/tmp/path.jsonl.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef union { int h264Type; int h265Type; } DataType;
typedef struct {
    uint32_t phyAddr, virAddr, length;
    int64_t timestamp;
    bool frameEnd;
    DataType dataType;
} EncoderPack;
typedef struct {
    EncoderPack *pack;
    uint32_t packCount, seq;
    int refType;
} EncoderStream;
typedef int (*GetStream)(int, EncoderStream *, bool);

#if defined(__mips__) && __SIZEOF_POINTER__ == 4
_Static_assert(sizeof(EncoderPack) == 32, "T23 pack size");
_Static_assert(offsetof(EncoderPack, timestamp) == 16, "T23 timestamp offset");
_Static_assert(offsetof(EncoderPack, frameEnd) == 24, "T23 frameEnd offset");
_Static_assert(offsetof(EncoderPack, dataType) == 28, "T23 type offset");
_Static_assert(sizeof(EncoderStream) == 16, "T23 stream size");
#endif

static GetStream original;
static int trace_fd = -1;
static unsigned counts[2];
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

__attribute__((constructor)) static void initialize(void) {
    original = (GetStream)dlsym(RTLD_NEXT, "IMP_Encoder_GetStream");
    const char *path = getenv("OPC_VIDEO_TRACE");
    if (original && path && path[0] == '/') {
        trace_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (trace_fd >= 0) {
            char line[256];
            int n = snprintf(line, sizeof(line),
                "{\"kind\":\"abi\",\"pack_size\":%zu,\"stream_size\":%zu,\"ts_offset\":%zu,\"frame_end_offset\":%zu}\n",
                sizeof(EncoderPack), sizeof(EncoderStream), offsetof(EncoderPack, timestamp),
                offsetof(EncoderPack, frameEnd));
            if (n > 0 && (size_t)n < sizeof(line)) (void)write(trace_fd, line, (size_t)n);
        }
    }
}

int IMP_Encoder_GetStream(int channel, EncoderStream *stream, bool block) {
    if (!original) original = (GetStream)dlsym(RTLD_NEXT, "IMP_Encoder_GetStream");
    if (!original) return -1;
    int result = original(channel, stream, block);
    if (result != 0 || trace_fd < 0 || channel < 0 || channel > 1 || !stream) return result;
    pthread_mutex_lock(&lock);
    if (counts[channel]++ < 400 && stream->pack && stream->packCount <= 64) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        for (uint32_t i = 0; i < stream->packCount; i++) {
            const EncoderPack *p = &stream->pack[i];
            char prefix[17] = "";
            const unsigned char *bytes = (const unsigned char *)(uintptr_t)p->virAddr;
            size_t sample = p->length < 8 ? p->length : 8;
            if (bytes) for (size_t j = 0; j < sample; j++) snprintf(prefix + j * 2, 3, "%02x", bytes[j]);
            char line[512];
            int n = snprintf(line, sizeof(line),
                "{\"kind\":\"pack\",\"channel\":%d,\"batch\":%u,\"seq\":%u,\"count\":%u,\"index\":%u,\"length\":%u,\"timestamp_us\":%lld,\"frame_end\":%s,\"sdk_type\":%d,\"prefix\":\"%s\",\"observed_ns\":%lld}\n",
                channel, counts[channel] - 1, stream->seq, stream->packCount, i, p->length,
                (long long)p->timestamp, p->frameEnd ? "true" : "false", p->dataType.h264Type,
                prefix, (long long)now.tv_sec * 1000000000LL + now.tv_nsec);
            if (n > 0 && (size_t)n < sizeof(line)) (void)write(trace_fd, line, (size_t)n);
        }
    }
    pthread_mutex_unlock(&lock);
    return result;
}
