/* Independent native caller with concurrent AI and AO threads. Never hardware. */
#define _POSIX_C_SOURCE 200809L
#include "imp_abi.h"
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int IMP_AI_GetPubAttr(int,OpcAudioAttr *);
int IMP_AO_GetPubAttr(int,OpcAudioAttr *);
int IMP_AO_EnableChn(int,int);
int IMP_AI_GetFrame(int,int,OpcAudioFrame *,int);
int IMP_AO_SendFrame(int,int,OpcAudioFrame *,int);
void opc_mock_stats(int *);
static volatile sig_atomic_t signal_stop;
static _Atomic int stopped;
static int fast;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9; }
static void stop(int s) { (void)s; signal_stop=1; }
static void pause_frame(void) { if(!fast) { struct timespec t={0,10000000}; nanosleep(&t,NULL); } }
static void *output(void *unused) {
    (void)unused;
    static unsigned int pcm[240];
    OpcAudioFrame f={0}; f.virAddr=pcm; f.len=960; f.bitwidth=16; f.soundmode=1;
    while(!atomic_load(&stopped)) { IMP_AO_SendFrame(0,0,&f,1); pause_frame(); }
    return NULL;
}
int main(int argc,char **argv) {
    double duration=argc>1?atof(argv[1]):1.;
    signal(SIGTERM,stop); signal(SIGINT,stop);
    fast=getenv("OPC_MOCK_FAST")!=NULL;
    OpcAudioAttr a;
    if(IMP_AI_GetPubAttr(0,&a) || IMP_AO_GetPubAttr(0,&a) || IMP_AO_EnableChn(0,0)) return 2;
    pthread_t thread;
    if(pthread_create(&thread,NULL,output,NULL)) return 3;
    double end=now()+duration; int failures=0;
    while(!signal_stop && now()<end) { OpcAudioFrame f; if(IMP_AI_GetFrame(0,0,&f,1)) ++failures; pause_frame(); }
    atomic_store(&stopped,1); pthread_join(thread,NULL);
    int counts[5];opc_mock_stats(counts);
    printf("{\"normal_gets\":%d,\"paired_gets\":%d,\"sends\":%d,\"enables\":%d,\"disables\":%d,\"get_failures\":%d}\n",counts[0],counts[1],counts[2],counts[3],counts[4],failures);
    return 0;
}
