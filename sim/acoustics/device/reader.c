/* Bounded supplementary observer. No IMP calls, no threads, no PCM alteration. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef MSG_DONTWAIT
/* macOS strict POSIX headers omit this; both descriptors are O_NONBLOCK. */
#define MSG_DONTWAIT 0
#endif
#define OPC_RING_BYTES (512U * 1024U)
#define OPC_CHUNK_BYTES (64U * 1024U)
static unsigned char ring[OPC_RING_BYTES];
static volatile sig_atomic_t interrupted;
static void on_signal(int sig) { interrupted=sig; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
int main(int argc,char **argv) {
    if(argc!=3) { fprintf(stderr,"usage: opc-audio-stream SOCKET SECONDS\n"); return 2; }
    char *end; double duration=strtod(argv[2],&end);
    if(*end || !isfinite(duration) || duration<=0 || duration>600) return 2;
    struct sockaddr_un a; memset(&a,0,sizeof(a)); a.sun_family=AF_UNIX;
    if(strlen(argv[1])>=sizeof(a.sun_path)) return 2;
    strcpy(a.sun_path,argv[1]);
    /* Fault in the fixed-capacity pages before accepting any producer burst. */
    volatile unsigned char *touch=ring;
    for(size_t i=0;i<sizeof(ring);i+=4096) touch[i]=0;
    touch[sizeof(ring)-1]=0;
    signal(SIGPIPE,SIG_IGN); signal(SIGTERM,on_signal); signal(SIGINT,on_signal);
    int fd=socket(AF_UNIX,SOCK_STREAM,0), rc=0, saved_errno=0;
    const char *reason="complete";
    size_t head=0,used=0,high=0;
    unsigned long long received=0,written=0,out_again=0,sock_again=0,partial=0;
    int requested=0,eof=0; double requested_at=0.,stop=0.,limit=0.;
    if(fd<0 || connect(fd,(struct sockaddr *)&a,sizeof(a))) { rc=3;saved_errno=errno;reason="socket_connect";goto done; }
    int flags=fcntl(fd,F_GETFL,0),out_flags=fcntl(STDOUT_FILENO,F_GETFL,0);
    if(flags<0 || out_flags<0 || fcntl(fd,F_SETFL,flags|O_NONBLOCK) || fcntl(STDOUT_FILENO,F_SETFL,out_flags|O_NONBLOCK)) {
        rc=10;saved_errno=errno;reason="nonblocking_setup";goto done;
    }
    stop=now()+duration;limit=stop+3.;
    for(;;) {
        int progress=0;
        double t=now();
        if(interrupted) { rc=128+interrupted;reason="interrupted";goto done; }
        /* Stop control is independent of stdout/ring backpressure. */
        if(!requested && !eof && t>=stop) {
            ssize_t n=send(fd,"S",1,MSG_DONTWAIT|MSG_NOSIGNAL);
            if(n==1) { requested=1;requested_at=t;progress=1; }
            else if(n<0 && (errno==EAGAIN || errno==EWOULDBLOCK || errno==EINTR)) { ++sock_again; }
            else { rc=4;saved_errno=n<0?errno:0;reason="stop_request";goto done; }
        }
        if(t>limit) { rc=used?11:5;reason=used?"stdout_drain_deadline":"producer_end_deadline";goto done; }
        if(used) {
            size_t n=used;
            if(n>sizeof(ring)-head) n=sizeof(ring)-head;
            if(n>OPC_CHUNK_BYTES) n=OPC_CHUNK_BYTES;
            ssize_t w=write(STDOUT_FILENO,ring+head,n);
            if(w>0) { if((size_t)w<n) ++partial;head=(head+(size_t)w)%sizeof(ring);used-=(size_t)w;written+=(size_t)w;progress=1; }
            else if(w<0 && (errno==EAGAIN || errno==EWOULDBLOCK)) { ++out_again; }
            else if(w<0 && errno==EINTR) { progress=1; }
            else { rc=9;saved_errno=w<0?errno:0;reason="stdout_write";goto done; }
        }
        if(!eof) {
            if(used==sizeof(ring)) {
                unsigned char byte;
                ssize_t n=recv(fd,&byte,1,MSG_PEEK|MSG_DONTWAIT);
                if(n>0) { rc=12;reason="ring_overflow";goto done; }
                if(n==0) { eof=1;progress=1;if(!requested && now()+3.<limit)limit=now()+3.; }
                else if(n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) { rc=8;saved_errno=errno;reason="socket_peek";goto done; }
            } else {
                size_t tail=(head+used)%sizeof(ring),space=sizeof(ring)-used;
                if(space>sizeof(ring)-tail) space=sizeof(ring)-tail;
                if(space>OPC_CHUNK_BYTES) space=OPC_CHUNK_BYTES;
                ssize_t n=read(fd,ring+tail,space);
                if(n>0) { used+=(size_t)n;received+=(size_t)n;if(used>high)high=used;progress=1; }
                else if(n==0) { eof=1;progress=1;if(!requested && now()+3.<limit)limit=now()+3.; }
                else if(errno==EAGAIN || errno==EWOULDBLOCK) { ++sock_again; }
                else if(errno==EINTR) { progress=1; }
                else { rc=8;saved_errno=errno;reason="socket_read";goto done; }
            }
        }
        if(eof && !used) { rc=requested?0:7;reason=requested?"complete":"unexpected_producer_eof";goto done; }
        if(progress) continue;
        struct pollfd p[2]={{eof?-1:fd,(short)(eof?0:POLLIN),0},{STDOUT_FILENO,(short)(used?POLLOUT:0),0}};
        if(!requested && !eof && now()>=stop)p[0].events|=POLLOUT;
        double next=requested?limit:stop;
        int wait_ms=(int)ceil((next-now())*1000.);
        if(wait_ms<1)wait_ms=1;
        if(wait_ms>50)wait_ms=50;
        if(poll(p,2,wait_ms)<0 && errno!=EINTR) { rc=6;saved_errno=errno;reason="poll";goto done; }
    }
 done:
    if(fd>=0)close(fd);
    char report[1024];
    int n=snprintf(report,sizeof(report),"{\"schema\":\"openpetcam.reader_io.v1\",\"exit_code\":%d,\"reason\":\"%s\",\"errno\":%d,\"ring_capacity_bytes\":%u,\"high_water_bytes\":%zu,\"received_bytes\":%llu,\"written_bytes\":%llu,\"stdout_eagain\":%llu,\"socket_eagain\":%llu,\"partial_writes\":%llu,\"stop_requested\":%s,\"stop_request_monotonic_s\":%.9f,\"producer_eof\":%s}\n",rc,reason,saved_errno,OPC_RING_BYTES,high,received,written,out_again,sock_again,partial,requested?"true":"false",requested_at,eof?"true":"false");
    if(n>0 && (size_t)n<sizeof(report)) { ssize_t ignored=write(STDERR_FILENO,report,(size_t)n);(void)ignored; }
    return rc;
}
