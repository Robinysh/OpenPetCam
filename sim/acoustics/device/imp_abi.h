/* T23 IMP audio ABI subset, from ingenic-headers T23/1.1.2/en/imp/imp_audio.h.
 * Deliberately no SDK initialization: the existing Prudynt owns both devices. */
#ifndef OPC_IMP_ABI_H
#define OPC_IMP_ABI_H
#include <stdint.h>
typedef struct {
    int bitwidth, soundmode;
    uint32_t *virAddr;
    uint32_t phyAddr;
    int64_t timeStamp;
    int seq, len;
} OpcAudioFrame;
typedef struct {
    int samplerate, bitwidth, soundmode, frmNum, numPerFrm, chnCnt;
} OpcAudioAttr;
typedef struct { int chnTotalNum, chnFreeNum, chnBusyNum; } OpcAudioQueue;
#if defined(__mips__) && __SIZEOF_POINTER__ == 4
_Static_assert(sizeof(OpcAudioFrame) == 32, "T23 o32 IMPAudioFrame ABI mismatch");
_Static_assert(__builtin_offsetof(OpcAudioFrame, timeStamp) == 16, "timestamp ABI mismatch");
#endif
#endif
