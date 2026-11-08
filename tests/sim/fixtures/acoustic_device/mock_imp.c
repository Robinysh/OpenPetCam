/* Native mock only: this is not an Ingenic SDK implementation or ABI proof. */
#include "imp_abi.h"
#include <stdlib.h>
#include <string.h>

static int gets, pairs, sends, enables, disables, sequence;
static unsigned int microphone[240], reference[240];
static const char *mode(void) { const char *s=getenv("OPC_MOCK_MODE"); return s?s:"valid"; }
static int frame(OpcAudioFrame *f) {
    memset(f,0,sizeof(*f));
    ++sequence;
    if(!strcmp(mode(),"sequence_gap") && sequence==10) ++sequence;
    f->bitwidth=16; f->soundmode=1; f->virAddr=microphone;
    f->seq=sequence; f->timeStamp=1000000LL+sequence*10000LL; f->len=960;
    for(unsigned i=0;i<240;i++) { microphone[i]=0x01000100U+(unsigned)sequence; reference[i]=0x02000200U; }
    return 0;
}
int IMP_AI_GetPubAttr(int d,OpcAudioAttr *a) { (void)d; memset(a,0,sizeof(*a)); a->samplerate=48000; a->bitwidth=16; a->soundmode=1; return 0; }
int IMP_AO_GetPubAttr(int d,OpcAudioAttr *a) { return IMP_AI_GetPubAttr(d,a); }
int IMP_AO_EnableChn(int d,int c) { (void)d; (void)c; return 0; }
int IMP_AI_GetFrame(int d,int c,OpcAudioFrame *f,int b) { (void)d;(void)c;(void)b; ++gets; return frame(f); }
int IMP_AI_GetFrameAndRef(int d,int c,OpcAudioFrame *f,OpcAudioFrame *r,int b) {
    (void)d;(void)c;(void)b; ++pairs;
    if(!strcmp(mode(),"pair_error")) return -17;
    frame(f); *r=*f; r->virAddr=reference;
    if(!strcmp(mode(),"malformed_reference")) r->len=958;
    return 0;
}
int IMP_AI_EnableAecRefFrame(int d,int c,int a,int b) { (void)d;(void)c;(void)a;(void)b; ++enables; return !strcmp(mode(),"reference_unavailable")?-1:0; }
int IMP_AI_DisableAecRefFrame(int d,int c,int a,int b) { (void)d;(void)c;(void)a;(void)b; ++disables; return 0; }
int IMP_AO_SendFrame(int d,int c,OpcAudioFrame *f,int b) { (void)d;(void)c;(void)f;(void)b; ++sends; return 0; }
int IMP_AO_QueryChnStat(int d,int c,OpcAudioQueue *q) { (void)d;(void)c; q->chnTotalNum=4; q->chnFreeNum=3; q->chnBusyNum=1; return 0; }
void opc_mock_stats(int *v) { v[0]=gets;v[1]=pairs;v[2]=sends;v[3]=enables;v[4]=disables; }
