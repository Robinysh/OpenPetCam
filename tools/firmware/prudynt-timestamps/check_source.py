#!/usr/bin/env python3
"""Build host checks against exact pinned/modified Prudynt source fragments.

No camera access or compiler/network discovery occurs. --baseline extracts the
old worker statements verbatim from the pinned Git object; the resulting failing
assertion is diagnostic evidence, not a certified full firmware build.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

PIN = '6afc4400c54a4739feeb72ac884587409577a2af'


def compile_run(source, include, compiler, directory):
    path = directory / 'check.cpp'
    path.write_text(source)
    binary = directory / 'check'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-Wno-unused-parameter', '-I', str(include), str(path), '-o', str(binary)], check=True)
    return subprocess.run([str(binary)], capture_output=True, text=True)


def baseline(tree, compiler, directory):
    src = subprocess.check_output(['git', '-C', str(tree), 'show', f'{PIN}:src/video/VideoWorker.cpp'], text=True)
    first = src[src.index('          bool frame_start = (i == 0)'):src.index('            // Peek ahead to detect')]
    second = src[src.index('          int64_t pack_ts_us = stream.pack[i].timestamp;'):src.index('          uint32_t h264_nal')]
    source = r'''
#include <cstdint>
#include <iostream>
#include <vector>
struct Pack { bool frameEnd; int64_t timestamp; int type; };
struct Stream { std::vector<Pack> pack; uint32_t packCount; };
int main() {
 int64_t ts_last_nonzero_us=0,ts_last_frame_us=0,ts_last_rtp_us=0,ts_current_frame_us=0;
 bool ts_have_current_frame=false,sei_pending_for_frame=false,sei_inserted_for_frame=false;
 const int64_t nominal_frame_step_us=40000;
 std::vector<int64_t> pictures;
 for(int n=0;n<750;n++) {
  const int64_t ts=1000000+n*40000LL;
  Stream stream;
  if(n%25==0) {stream.pack.push_back({true,ts,7});stream.pack.push_back({false,ts,8});}
  stream.pack.push_back({true,ts,1});stream.packCount=stream.pack.size();
  for(uint32_t i=0;i<stream.packCount;i++) {
   const bool pack_is_frame_end=stream.pack[i].frameEnd||(i+1==stream.packCount);
''' + first + '\n          }\n' + second + r'''
   if(stream.pack[i].type==1) pictures.push_back(rtsp_ts_us);
  }
 }
 (void)sei_pending_for_frame;(void)sei_inserted_for_frame;
 const int64_t actual=pictures.back()-pictures.front(),expected=29960000;
 std::cout<<"750 actual pictures: expected span "<<expected<<"us, observed "<<actual<<"us\n";
 if(actual!=expected) {std::cerr<<"ASSERTION FAILED: config packs must not consume picture periods\n";return 1;}
 return 0;
}
'''
    return compile_run(source, tree / 'src/video', compiler, directory)


def patched(tree, compiler, directory):
    src = (tree / 'src/network/rtsp/RtspServer.cpp').read_text()
    start = src.index('void RtspServer::updateVideoTimestamp(')
    finish = src.index('// -- RTP sending (video)', start)
    method = src[start:finish]
    source = r'''
#include "AccessUnitTiming.hpp"
#include <cassert>
#include <cstdint>
#include <ctime>
#include <iostream>
struct H264NALUnit {bool is_frame_start=false;int64_t imp_ts=0;};
struct Rtp {uint32_t timestamp=0;};
struct Session {int64_t videoStartAnchorUs=-1,videoTsToMonoOffset=0,lastVideoTsUs=-1;
 uint32_t lastFrameRtpTs=0,videoFrameCount=0;bool hasFrameRtpTs=false;Rtp videoRtp;};
class RtspServer {public:void updateVideoTimestamp(Session&,const H264NALUnit&,const uint8_t*,size_t,bool);};
''' + method + r'''
int main() {
 RtspServer server;Session s;
 const uint8_t prefix=7,idr=5,slice=1;
 for(int n=0;n<750;n++) {
  const int64_t capture=1000000+n*40000LL;
  H264NALUnit first{true,capture};
  server.updateVideoTimestamp(s,first,n%25==0?&prefix:&slice,1,false);
  H264NALUnit more{false,capture};
  server.updateVideoTimestamp(s,more,&idr,1,false);
  assert(s.videoRtp.timestamp==static_cast<uint32_t>(n*3600));
 }
 assert(s.videoFrameCount==750 && s.videoRtp.timestamp==2696400);
 const int64_t mono=s.lastVideoTsUs;
 server.updateVideoTimestamp(s,{false,0},&prefix,1,false);
 assert(s.lastVideoTsUs==mono && s.videoFrameCount==750);
 // This compiles the real patched RTSP method, not a replacement clock model.
 std::cout<<"Real patched RtspServer::updateVideoTimestamp: 750 pictures, config neutral, 29.96s span\n";
}
'''
    return compile_run(source, tree / 'src/video', compiler, directory)


def worker(tree, compiler, directory):
    src = (tree / 'src/video/VideoWorker.cpp').read_text()
    start = src.index('        // Group pictures before touching clock state.')
    finish = src.index('          bool recorder_active', start)
    statements = src[start:finish]
    # Compile the exact production grouping/copy/clock path with a bounded fake
    # SDK data carrier. The same header is consumed by the full target build.
    source = r"""
#include "AccessUnitTiming.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
struct FakePack {int64_t timestamp;bool frameEnd;std::vector<uint8_t> bytes;};
struct FakeStream {std::vector<FakePack> pack;uint32_t packCount;};
namespace hal { namespace encoder {
struct Slices {uint8_t *first_ptr;uint32_t first_len;uint8_t *second_ptr;uint32_t second_len;};
Slices get_pack_slices(FakeStream &s,uint32_t j) {
 return {s.pack[j].bytes.data(),static_cast<uint32_t>(s.pack[j].bytes.size()),nullptr,0};
}
}}
#define LOG_WARN(message) do {std::ostringstream text;text<<message;} while(0)
static int idr_requests=0;
static void IMP_Encoder_RequestIDR(int) {idr_requests++;}
static FakePack make_pack(int type,int64_t ts,bool first,bool end) {
 return {ts,end,{0,0,0,1,static_cast<uint8_t>(type),static_cast<uint8_t>(first?0x80:0x40)}};
}
int main() {
 access_unit_timing::PictureClock picture_clock;
 const int encChn=0;const bool stream_is_h265=false;
 size_t picture_count=0,end_count=0,nal_count=0;
 for(int n=0;n<750;n++) {
  FakeStream stream;
  const int64_t ts=1000000+n*40000LL;
  if(n%25==0) {
   stream.pack.push_back(make_pack(7,ts,false,true));
   stream.pack.push_back(make_pack(8,ts,false,false));
  }
  stream.pack.push_back(make_pack(n%25==0?5:1,ts,true,true));
  if(n%3==0) stream.pack.push_back(make_pack(1,ts,false,true));
  stream.packCount=stream.pack.size();
""" + statements + r"""
   assert(pack_ts_us==ts && rtsp_ts_us==ts);
   if(frame_start) picture_count++;
   if(pack_is_frame_end) end_count++;
   nal_count++;
  }
 }
 assert(picture_count==750 && end_count==750 && nal_count==1060 && idr_requests==0);
 assert(picture_clock.last()==30960000);
 std::cout<<"Exact patched VideoWorker grouping/copy/clock statements:750 pictures/1060 NALs,750 markers; no config or multislice advance\n";
}
"""
    return compile_run(source, tree / 'src/video', compiler, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tree', type=Path)
    parser.add_argument('--compiler', default='c++')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--baseline', action='store_true')
    mode.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='opc-prudynt-clock-') as tmp:
        result = (baseline if args.baseline else worker if args.worker else patched)(args.tree, args.compiler, Path(tmp))
    print(result.stdout, end='')
    print(result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
