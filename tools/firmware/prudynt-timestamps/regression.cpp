#include "AccessUnitTiming.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>
using namespace access_unit_timing;

static Pack config(int type, int64_t ts = 0) {
    uint8_t payload[] = {0, 0, 0, 1, static_cast<uint8_t>(type), 0x80};
    return describe(payload, sizeof(payload), false, ts);
}
static Pack picture(int64_t ts, bool first = true) {
    uint8_t payload[] = {0, 0, 0, 1, 0x65, static_cast<uint8_t>(first ? 0x80 : 0x40)};
    return describe(payload, sizeof(payload), false, ts);
}
static std::vector<int64_t> sequence(int gop) {
    PictureClock clock;
    std::vector<int64_t> stamps;
    for (int n = 0; n < 750; ++n) {
        int64_t ts = 1000000 + n * 40000LL;
        std::vector<Pack> packs;
        if (n % gop == 0) {
            packs.push_back(config(7, ts)); packs.push_back(config(8, ts));
        }
        packs.push_back(picture(ts));
        const auto units = group(packs);
        assert(units.size() == 1 && units[0].picture && units[0].valid);
        const auto &u = units[0];
        assert(clock.accept(u.timestamp_us) == Result::Accepted);
        for (size_t j = u.begin; j <= u.end; ++j) {
            // The production worker assigns this single time to every NAL.
            assert(u.timestamp_us == ts);
        }
        stamps.push_back(u.timestamp_us);
    }
    assert(stamps.size() == 750);
    assert(stamps.back() - stamps.front() == 29960000);
    assert(rtp90k(stamps.back(), stamps.front()) == 2696400);
    return stamps;
}
int main() {
    const auto a = sequence(10), b = sequence(25), c = sequence(50);
    assert(a == b && b == c);
    {
        PictureClock clock;
        auto only_config = group({config(7), config(8), config(6)});
        assert(only_config.size() == 1 && !only_config[0].picture);
        assert(clock.last() == 0);
        auto units = group({config(7), config(7), config(8), config(6),
                            picture(50000), picture(50000, false),
                            picture(90000), picture(90000, false)});
        assert(units.size() == 2 && units[0].begin == 0 && units[0].end == 5);
        assert(units[1].begin == 6 && units[1].end == 7);
        assert(units[0].valid && units[1].valid);
        assert(clock.accept(units[0].timestamp_us) == Result::Accepted);
        assert(clock.accept(units[1].timestamp_us) == Result::Accepted);
    }
    {
        auto units = group({picture(100000), config(7), config(8), picture(140000)});
        assert(units.size() == 2 && units[0].end == 0 && units[1].begin == 1);
        assert(group({picture(100000, false)})[0].valid == false);
        assert(group({picture(100000), picture(100001, false)})[0].valid == false);
        Pack bad; bad.valid = false;
        assert(!group({bad, picture(100000)})[0].valid);
    }
    {
        PictureClock clock;
        assert(clock.accept(0) == Result::Invalid && clock.last() == 0);
        assert(clock.accept(1000000) == Result::Accepted);
        assert(clock.accept(1000000) == Result::Repeated && clock.last() == 1000000);
        assert(clock.accept(900000) == Result::Backward && clock.last() == 1000000);
        assert(clock.accept(1040000) == Result::Accepted);
        assert(clock.accept(1200000) == Result::Accepted); // preserve actual 160ms capture gap
        assert(rtp90k(1200000, 1040000) == 14400);
        assert(clock.accept(2200000) == Result::Accepted); // no 500ms clamp
        assert(rtp90k(2200000, 1200000) == 90000);
        clock.reset();
        assert(clock.accept(5000) == Result::Accepted); // explicit source restart establishes new epoch
    }
    {
        constexpr int64_t anchor = 1000000;
        constexpr int64_t near_wrap = 47721840000LL;
        const uint32_t before = rtp90k(anchor + near_wrap, anchor);
        const uint32_t after = rtp90k(anchor + near_wrap + 40000, anchor);
        assert(after < before && static_cast<uint32_t>(after - before) == 3600);
        assert(rtp90k(anchor, anchor) == rtp90k(anchor, anchor));
        assert(rtp90k(anchor + 40000, anchor) == 3600);
    }
    {
        uint8_t h265_first[] = {0,0,0,1,static_cast<uint8_t>(19 << 1),1,0x80};
        uint8_t h265_more[] = {0,0,0,1,static_cast<uint8_t>(19 << 1),1,0};
        auto first = describe(h265_first, sizeof(h265_first), true, 40000);
        auto more = describe(h265_more, sizeof(h265_more), true, 40000);
        assert(first.picture && first.first_slice && more.picture && !more.first_slice);
        assert(group({first, more}).size() == 1);
        assert(!describe(nullptr, 8, false, 0).valid);
        uint8_t truncated[] = {0,0,0,1,0x65};
        assert(!describe(truncated, sizeof(truncated), false, 40000).valid);
    }
    std::cout << "750 pictures, 29.96s span; GOP10/25/50 identical; config/multislice/gap/discontinuity/wrap checks passed\n";
}
