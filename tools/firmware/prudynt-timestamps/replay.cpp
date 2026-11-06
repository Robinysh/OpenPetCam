#include "AccessUnitTiming.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
struct Observed {int channel,batch,index,count,type;int64_t timestamp;bool end;std::string prefix;};
int main() {
    std::map<std::pair<int,int>,std::vector<Observed>> batches;
    Observed p;
    while(std::cin>>p.channel>>p.batch>>p.index>>p.count>>p.timestamp>>p.end>>p.type>>p.prefix)
        batches[{p.channel,p.batch}].push_back(p);
    std::array<access_unit_timing::PictureClock,2> clocks;
    std::array<int64_t,2> anchors{};
    for(const auto &entry:batches) {
        const auto &rows=entry.second;
        const int ch=entry.first.first;
        if(ch<0||ch>1) return 2;
        std::vector<access_unit_timing::Pack> packs;
        for(const auto &row:rows) {
            std::vector<uint8_t> bytes;
            for(size_t i=0;i+1<row.prefix.size();i+=2)
                bytes.push_back(static_cast<uint8_t>(std::stoul(row.prefix.substr(i,2),nullptr,16)));
            packs.push_back(access_unit_timing::describe(bytes.data(),bytes.size(),false,row.timestamp));
        }
        const auto units=access_unit_timing::group(packs);
        for(const auto &unit:units) {
            auto result=access_unit_timing::Result::Accepted;
            if(unit.picture) result=clocks[ch].accept(unit.timestamp_us);
            if(unit.picture&&!anchors[ch]) anchors[ch]=unit.timestamp_us;
            const int64_t assigned=unit.picture?unit.timestamp_us:clocks[ch].last();
            for(size_t i=unit.begin;i<=unit.end;i++) {
                const auto &row=rows[i];
                std::cout<<"{\"channel\":"<<ch<<",\"batch\":"<<row.batch<<",\"index\":"<<row.index
                    <<",\"input_us\":"<<row.timestamp<<",\"assigned_us\":"<<assigned
                    <<",\"frame_start\":"<<(unit.picture&&i==unit.begin?"true":"false")
                    <<",\"frame_end\":"<<(unit.picture&&i==unit.end?"true":"false")
                    <<",\"has_picture\":"<<(unit.picture?"true":"false")
                    <<",\"valid\":"<<(unit.valid?"true":"false")
                    <<",\"clock_result\":"<<static_cast<int>(result)
                    <<",\"rtp90k\":"<<access_unit_timing::rtp90k(assigned,anchors[ch])<<"}\n";
            }
        }
    }
}
