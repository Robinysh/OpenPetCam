/* Opt-in, fail-closed audio observation for Prudynt stable 6afc440.
 * AI is never opened here: interpose the streamer's existing GetFrame call.
 * SDK output reference is NOT claimed to be physical DAC feedback.
 * Native tests use the same code with a mock libimp; deployment needs review. */
#define _GNU_SOURCE
#include "imp_abi.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h> /* chmod declaration only: no cross-libc struct-stat use. */
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__) && __SIZEOF_POINTER__ == 4
/* musl's time64 redirect is an ABI adapter, not present in host uClibc. We
 * resolve only IMP symbols, so explicitly use the host's ordinary dlsym. */
extern void *opc_host_dlsym(void *, const char *) __asm__("dlsym");
#define dlsym opc_host_dlsym
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define OPC_MAX_FRAME (256 * 1024)
#define OPC_REV "prudynt-6afc440-t23-abi1-refgrid2-buffered-io1"
#define OPC_REQUEST_SNDBUF (256 * 1024)
#define OPC_MAX_SNDBUF (1024 * 1024)
#define OPC_DIAGNOSTIC_BUDGET (64 * 1024)
static int listener = -1, client = -1, enabled, want_ref, ref_state;
static int ao_dev, ao_ch, ao_ready;
static OpcAudioAttr ai_attr, ao_attr;
static unsigned long long stream_id, index_no, drops, ao_index;
static volatile int gate;
static char socket_path[sizeof(((struct sockaddr_un *)0)->sun_path)];
static uint32_t crc_table[256];
static int diagnostic_fd=-1, client_sndbuf;
static size_t diagnostic_remaining;
static int (*real_get)(int,int,OpcAudioFrame *,int);
static int (*real_pair)(int,int,OpcAudioFrame *,OpcAudioFrame *,int);
static int (*enable_ref)(int,int,int,int);
static int (*disable_ref)(int,int,int,int);
static int (*real_send)(int,int,OpcAudioFrame *,int);
static int (*real_ai_attr)(int,OpcAudioAttr *);
static int (*real_ao_attr)(int,OpcAudioAttr *);
static int (*real_ao_enable)(int,int);
static int (*query_ao)(int,int,OpcAudioQueue *);

static void lock_gate(void) {
    while (__sync_lock_test_and_set(&gate, 1)) sched_yield();
}
static void unlock_gate(void) { __sync_lock_release(&gate); }
static unsigned long long now_ns(void) {
#if defined(__linux__) && __SIZEOF_POINTER__ == 4
    /* Avoid a time_t64-musl/time_t32-uClibc struct mismatch in this preload. */
    struct { long sec, nsec; } t;
#ifdef SYS_clock_gettime32
    if (syscall(SYS_clock_gettime32, CLOCK_MONOTONIC, &t)) return 0;
#else
    if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &t)) return 0;
#endif
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
#define sec tv_sec
#define nsec tv_nsec
#endif
    unsigned long long n = (unsigned long long)t.sec * 1000000000ULL + t.nsec;
#if !defined(__linux__) || __SIZEOF_POINTER__ != 4
#undef sec
#undef nsec
#endif
    return n;
}
/* Optional fresh /tmp file, bounded no-PCM evidence. One best-effort write. */
static void diagnostic(const char *event, int saved_errno, long long sent, size_t expected, int set_error) {
    if(diagnostic_fd<0 || !diagnostic_remaining) return;
    char line[768];
    int n=snprintf(line,sizeof(line),"{\"schema\":\"openpetcam.tap_io.v1\",\"event\":\"%s\",\"pid\":%ld,\"monotonic_ns\":%llu,\"stream_id\":\"%llu\",\"record_index\":%llu,\"drop_counter_before\":%llu,\"errno\":%d,\"sent_bytes\":%lld,\"expected_bytes\":%zu,\"sndbuf_requested\":%d,\"sndbuf_actual\":%d,\"setsockopt_errno\":%d}\n",
        event,(long)getpid(),now_ns(),stream_id,index_no,drops,saved_errno,sent,expected,OPC_REQUEST_SNDBUF,client_sndbuf,set_error);
    if(n<=0 || (size_t)n>=sizeof(line) || (size_t)n>diagnostic_remaining) return;
    ssize_t written=write(diagnostic_fd,line,(size_t)n);
    if(written>0) diagnostic_remaining-=(size_t)written;
    /* A failed diagnostic never changes normal SDK operation or retries I/O. */
}
static void setup_diagnostic(void) {
    const char *path=getenv("OPC_AUDIO_DIAGNOSTIC_PATH");
    if(!path || strncmp(path,"/tmp/",5) || strlen(path)>200 || !path[5]) return;
    /* Direct basename only; no intermediate links, traversal, or existing file. */
    for(const char *p=path+5;*p;++p)
        if(!((*p>='a'&&*p<='z') || (*p>='A'&&*p<='Z') || (*p>='0'&&*p<='9') || *p=='-' || *p=='_' || *p=='.')) return;
    if(!strcmp(path+5,".") || !strcmp(path+5,"..")) return;
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0) return;
    diagnostic_fd=fd;diagnostic_remaining=OPC_DIAGNOSTIC_BUDGET;
}
static uint32_t crc_update(uint32_t c, const void *ptr, size_t n) {
    const unsigned char *p = ptr;
    while (n--) c = crc_table[(c ^ *p++) & 255] ^ (c >> 8);
    return c;
}
static void le32(unsigned char *p, uint32_t v) {
    for (int j = 0; j < 4; ++j) p[j] = (unsigned char)(v >> (8*j));
}
static void disconnect_bad(void) {
    ++drops;
    if (client >= 0) close(client);
    client = -1;
}
static int send_record(const char *json, const void *a, size_t na, const void *b, size_t nb) {
    if (client < 0) return 0;
    size_t nj = strlen(json);
    if (nj > 8192 || na + nb > OPC_MAX_FRAME) { diagnostic("frame_bounds",EMSGSIZE,-1,20+nj+na+nb,0); disconnect_bad(); return 0; }
    unsigned char prefix[20] = "OPCAUD01";
    le32(prefix+8, (uint32_t)nj); le32(prefix+12, (uint32_t)(na+nb));
    uint32_t crc = crc_update(0xffffffffU, json, nj);
    crc = crc_update(crc, a, na); crc = crc_update(crc, b, nb);
    le32(prefix+16, crc ^ 0xffffffffU);
    struct iovec v[4] = {{prefix,sizeof(prefix)}, {(void *)json,nj}, {(void *)a,na}, {(void *)b,nb}};
    struct msghdr msg; memset(&msg, 0, sizeof(msg)); msg.msg_iov=v; msg.msg_iovlen=4;
    ssize_t sent = sendmsg(client, &msg, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent != (ssize_t)(20+nj+na+nb)) { int failure_errno=sent<0?errno:0; diagnostic("send_failed",failure_errno,(long long)sent,20+nj+na+nb,0); disconnect_bad(); return 0; }
    ++index_no;
    return 1;
}
static void control_record(const char *kind) {
    char j[1500];
    snprintf(j,sizeof(j),"{\"schema\":\"openpetcam.audio.v1\",\"kind\":\"%s\","
       "\"stream_id\":\"%llu\",\"record_index\":%llu,\"dropped_records\":%llu,"
       "\"producer_pid\":%ld,\"monotonic_ns\":%llu,\"reference_requested\":%s,"
       "\"reference_state\":%d,\"sdk_timestamp_semantics\":\"unverified_block_boundary\","
       "\"reference_kind\":\"sdk_output_reference_not_validated_physical_DAC\","
       "\"source_pin\":\"%s\"}",kind,stream_id,index_no,drops,(long)getpid(),now_ns(),
       want_ref?"true":"false",ref_state,OPC_REV);
    send_record(j,NULL,0,NULL,0);
}
static void service(void) {
    if (!enabled) return;
    if (client < 0) {
        int c = accept(listener,NULL,NULL);
        if (c >= 0) {
#ifdef SO_NOSIGPIPE
            int one=1; setsockopt(c,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
#endif
            client=c; index_no=0; stream_id=now_ns(); client_sndbuf=0;
            int wanted=OPC_REQUEST_SNDBUF;
            int set_error=setsockopt(c,SOL_SOCKET,SO_SNDBUF,&wanted,sizeof(wanted))<0?errno:0;
            socklen_t length=sizeof(client_sndbuf);
            int get_error=getsockopt(c,SOL_SOCKET,SO_SNDBUF,&client_sndbuf,&length)<0?errno:0;
            int flags=fcntl(c,F_GETFL,0);
            int flag_error=flags<0?errno:(fcntl(c,F_SETFL,flags|O_NONBLOCK)<0?errno:0);
            if(set_error || get_error || flag_error || client_sndbuf<=0 || client_sndbuf>OPC_MAX_SNDBUF) {
                diagnostic("socket_setup_failed",get_error?get_error:(flag_error?flag_error:set_error),-1,0,set_error);
                disconnect_bad();return;
            }
            diagnostic("socket_accepted",0,0,0,0);
            control_record("hello");
        }
    }
    if (client >= 0) {
        char c; ssize_t n=recv(client,&c,1,MSG_DONTWAIT);
        if (n == 0) { close(client); client=-1; }
        else if (n == 1 && c == 'S') {
            control_record("end");
            if (client>=0) close(client);
            client=-1;
        }
    }
}
static int active(void) {
    lock_gate(); service(); int yes=client>=0; unlock_gate(); return yes;
}
static void publish_mic(const OpcAudioFrame *m,const OpcAudioFrame *r,
                        unsigned long long before,unsigned long long after,int rc) {
    lock_gate(); service();
    if (client < 0) { unlock_gate(); return; }
    char j[2300];
    int mr = m && m->len>0 && m->len<=OPC_MAX_FRAME && m->virAddr;
    int rr = r && r->len>0 && r->len<=OPC_MAX_FRAME && r->virAddr;
    snprintf(j,sizeof(j),"{\"schema\":\"openpetcam.audio.v1\",\"kind\":\"%s\","
      "\"stream_id\":\"%llu\",\"record_index\":%llu,\"dropped_records\":%llu,"
      "\"monotonic_before_ns\":%llu,\"monotonic_after_ns\":%llu,\"rc\":%d,\"reference_state\":%d,"
      "\"mic\":{\"seq\":%d,\"timestamp_us\":%lld,\"bytes\":%d,\"rate_hz\":%d,\"bitwidth\":%d,\"channels\":%d},"
      "\"reference\":{\"seq\":%d,\"timestamp_us\":%lld,\"bytes\":%d,\"rate_hz\":%d,\"rate_basis\":\"paired_AI_grid_candidate_requires_active_validation\",\"ao_input_rate_hz\":%d,\"bitwidth\":%d,\"channels\":%d}}",
      rc || !mr?"fault":"mic",stream_id,index_no,drops,before,after,rc,ref_state,
      mr?m->seq:0,mr?(long long)m->timeStamp:0,mr?m->len:0,ai_attr.samplerate,mr?m->bitwidth:0,mr?m->soundmode:0,
      rr?r->seq:0,rr?(long long)r->timeStamp:0,rr?r->len:0,ai_attr.samplerate,ao_attr.samplerate,rr?r->bitwidth:0,rr?r->soundmode:0);
    send_record(j,mr?m->virAddr:NULL,mr?(size_t)m->len:0,rr?r->virAddr:NULL,rr?(size_t)r->len:0);
    unlock_gate();
}
__attribute__((constructor)) static void setup(void) {
    const char *p=getenv("OPC_AUDIO_SOCKET");
#define RESOLVE(dst,name) *(void **)(&dst)=dlsym(RTLD_NEXT,name)
    RESOLVE(real_get,"IMP_AI_GetFrame"); RESOLVE(real_pair,"IMP_AI_GetFrameAndRef");
    RESOLVE(enable_ref,"IMP_AI_EnableAecRefFrame"); RESOLVE(disable_ref,"IMP_AI_DisableAecRefFrame");
    RESOLVE(real_send,"IMP_AO_SendFrame"); RESOLVE(real_ai_attr,"IMP_AI_GetPubAttr");
    RESOLVE(real_ao_attr,"IMP_AO_GetPubAttr"); RESOLVE(real_ao_enable,"IMP_AO_EnableChn");
    RESOLVE(query_ao,"IMP_AO_QueryChnStat");
#undef RESOLVE
    for (unsigned i=0;i<256;i++) { uint32_t c=i; for(int k=0;k<8;k++) c=(c>>1)^((c&1)?0xedb88320U:0); crc_table[i]=c; }
    /* Passthrough wrappers must resolve even when observation is disabled. */
    if (!p || !*p || strlen(p)>=sizeof(socket_path) || !real_get) return;
    setup_diagnostic();
    const char *ref=getenv("OPC_AUDIO_REFERENCE"); want_ref=ref && !strcmp(ref,"1");
    struct sockaddr_un addr; memset(&addr,0,sizeof(addr)); addr.sun_family=AF_UNIX;
    strcpy(addr.sun_path,p); strcpy(socket_path,p);
    /* Refuse to remove an existing endpoint: stale/other sessions need review. */
    listener=socket(AF_UNIX,SOCK_STREAM,0);
    if(listener<0) return;
    if(bind(listener,(struct sockaddr *)&addr,sizeof(addr)) || listen(listener,1)) { close(listener); listener=-1; return; }
    chmod(p,0600); fcntl(listener,F_SETFL,fcntl(listener,F_GETFL,0)|O_NONBLOCK); enabled=1;
}
__attribute__((destructor)) static void teardown(void) {
    if(client>=0) close(client);
    if(listener>=0) { close(listener); unlink(socket_path); }
    if(diagnostic_fd>=0) close(diagnostic_fd);
}
int IMP_AI_GetPubAttr(int d,OpcAudioAttr *a) {
    int rc=real_ai_attr?real_ai_attr(d,a):-1;
    if(!rc && a) { lock_gate(); ai_attr=*a; unlock_gate(); } return rc;
}
int IMP_AO_GetPubAttr(int d,OpcAudioAttr *a) {
    int rc=real_ao_attr?real_ao_attr(d,a):-1;
    if(!rc && a) { lock_gate(); ao_attr=*a; unlock_gate(); } return rc;
}
int IMP_AO_EnableChn(int d,int c) {
    int rc=real_ao_enable?real_ao_enable(d,c):-1;
    if(!rc) { lock_gate(); ao_dev=d; ao_ch=c; ao_ready=1; unlock_gate(); } return rc;
}
int IMP_AI_GetFrame(int d,int c,OpcAudioFrame *m,int block) {
    int recording=active();
    lock_gate(); int state=ref_state, od=ao_dev, oc=ao_ch, ready=ao_ready; unlock_gate();
    if(!recording && state==1 && disable_ref) {
        disable_ref(d,c,od,oc); state=0;
        lock_gate(); ref_state=state; unlock_gate();
    }
    if(recording) {
        OpcAudioAttr attr;
        if(real_ai_attr && !real_ai_attr(d,&attr)) { lock_gate(); ai_attr=attr; unlock_gate(); }
        if(real_ao_attr && ready && !real_ao_attr(od,&attr)) { lock_gate(); ao_attr=attr; unlock_gate(); }
    }
    if(recording && want_ref && state==0 && ready) {
        state=enable_ref && real_pair && !enable_ref(d,c,od,oc)?1:-1;
        lock_gate(); ref_state=state; unlock_gate();
    }
    unsigned long long before=now_ns();
    OpcAudioFrame ref; memset(&ref,0,sizeof(ref));
    int rc;
    if(state==1) {
        rc=real_pair(d,c,m,&ref,block);
        if(rc) {
            lock_gate(); ref_state=-2; unlock_gate();
            if(disable_ref) disable_ref(d,c,od,oc);
        }
    } else rc=real_get?real_get(d,c,m,block):-1;
    unsigned long long after=now_ns();
    if(recording) publish_mic(rc?NULL:m,&ref,before,after,rc);
    return rc;
}
int IMP_AO_SendFrame(int d,int c,OpcAudioFrame *f,int block) {
    OpcAudioQueue a={-1,-1,-1},b={-1,-1,-1};
    int recording=active();
    if(recording && real_ao_attr) {
        OpcAudioAttr attr;
        if(!real_ao_attr(d,&attr)) { lock_gate(); ao_attr=attr; unlock_gate(); }
    }
    if(recording && query_ao) query_ao(d,c,&a);
    unsigned long long before=now_ns();
    int rc=real_send?real_send(d,c,f,block):-1;
    unsigned long long after=now_ns();
    if(recording && query_ao) query_ao(d,c,&b);
    if(recording) {
        lock_gate(); service();
        char j[1600]; int valid=f && f->virAddr && f->len>0 && f->len<=OPC_MAX_FRAME;
        snprintf(j,sizeof(j),"{\"schema\":\"openpetcam.audio.v1\",\"kind\":\"%s\","
           "\"stream_id\":\"%llu\",\"record_index\":%llu,\"dropped_records\":%llu,"
           "\"monotonic_before_ns\":%llu,\"monotonic_after_ns\":%llu,\"rc\":%d,"
           "\"ao\":{\"submission_index\":%llu,\"bytes\":%d,\"rate_hz\":%d,\"bitwidth\":%d,\"channels\":%d,"
           "\"busy_blocks_before\":%d,\"busy_blocks_after\":%d,\"timing\":\"submission_not_DAC_emission\"}}",
           rc || !valid?"fault":"ao",stream_id,index_no,drops,before,after,rc,ao_index++,
           valid?f->len:0,ao_attr.samplerate,ao_attr.bitwidth,ao_attr.soundmode,a.chnBusyNum,b.chnBusyNum);
        send_record(j,valid?f->virAddr:NULL,valid?(size_t)f->len:0,NULL,0);
        unlock_gate();
    }
    return rc;
}
