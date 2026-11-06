// Scalar 3-D wave equation. Compiled only by wave3d.py into an explicit cache.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

template<class T> T read(std::ifstream& f) {
    T value; f.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("truncated input");
    return value;
}
template<class T> void read_array(std::ifstream& f, std::vector<T>& a) {
    f.read(reinterpret_cast<char*>(a.data()), static_cast<std::streamsize>(a.size()*sizeof(T)));
    if (!f) throw std::runtime_error("truncated array");
}
template<class T> void write(std::ofstream& f, const T& value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("output failed");
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("requires input and output paths");
    std::ifstream in(argv[1], std::ios::binary);
    char magic[8]; in.read(magic, 8);
    if (!in || std::memcmp(magic, "OPCW3D1\0", 8)) throw std::runtime_error("bad input protocol");
    const uint32_t nx=read<uint32_t>(in), ny=read<uint32_t>(in), nz=read<uint32_t>(in);
    const uint64_t nt=read<uint64_t>(in), nr=read<uint64_t>(in), ns=read<uint64_t>(in);
    const double h=read<double>(in), c=read<double>(in), dt=read<double>(in);
    const uint32_t stride=read<uint32_t>(in);
    const uint64_t cells=uint64_t(nx)*ny*nz, yz=uint64_t(ny)*nz;
    if (nx<3 || ny<3 || nz<3 || cells>8000000 || nt<2 || nt>100000 || !nr || nr>10000 ||
        nr*nt>64000000 || !ns || ns>cells || !std::isfinite(h) || h<=0 || !std::isfinite(c) || c<=0 ||
        !std::isfinite(dt) || dt<=0 || c*dt/h>1/std::sqrt(3.0)+1e-14)
        throw std::runtime_error("invalid grid/resource/CFL parameters");
    std::vector<uint8_t> solid(cells);
    std::vector<double> sigma(cells), drive(nt), sw(ns), rw(nr*8);
    std::vector<uint64_t> si(ns), ri(nr*8);
    read_array(in,solid); read_array(in,sigma); read_array(in,drive);
    read_array(in,si); read_array(in,sw); read_array(in,ri); read_array(in,rw);
    if (in.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("unexpected input bytes");
    for (uint64_t i=0;i<cells;++i) if (solid[i]>1 || !std::isfinite(sigma[i]) || sigma[i]<0 ||
        !std::isfinite(sigma[i]*dt)) throw std::runtime_error("invalid material arrays");
    for (double value:drive) if (!std::isfinite(value)) throw std::runtime_error("invalid drive");
    for (uint64_t i=0;i<ns;++i) if (si[i]>=cells || solid[si[i]] || !std::isfinite(sw[i]))
        throw std::runtime_error("invalid source stencil");
    for (uint64_t i=0;i<nr*8;++i) if (ri[i]>=cells || !std::isfinite(rw[i]) || rw[i]<0 ||
        (rw[i]>1e-14 && solid[ri[i]])) throw std::runtime_error("invalid receiver stencil");
    const uint64_t ne=stride ? (nt-2)/stride+1 : 0;
    std::ofstream out(argv[2],std::ios::binary);
    out.write("OPCWOUT1",8); write(out,nt); write(out,nr); write(out,ne); write(out,dt);
    std::vector<double> old(cells,0), current(cells,0), next(cells,0), probes(nr), et, en;
    const double r2=(c*dt/h)*(c*dt/h), source_scale=dt*dt/(h*h*h);
    for (uint64_t step=0;step<nt;++step) {
      for (uint64_t k=0;k<nr;++k) {
        double value=0;
        for (uint64_t a=0;a<8;++a) value+=rw[k*8+a]*current[ri[k*8+a]];
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite receiver pressure");
        probes[k]=value;
      }
      out.write(reinterpret_cast<const char*>(probes.data()), static_cast<std::streamsize>(nr*sizeof(double)));
      if (!out) throw std::runtime_error("signal output failed");
      if (step+1==nt) break;
      for (uint32_t x=0;x<nx;++x) for (uint32_t y=0;y<ny;++y) for (uint32_t z=0;z<nz;++z) {
        const uint64_t i=uint64_t(x)*yz+uint64_t(y)*nz+z;
        if (solid[i]) {next[i]=0; continue;}
        const double p=current[i]; double lap=0;
        if (x && !solid[i-yz]) lap+=current[i-yz]-p;
        if (x+1<nx && !solid[i+yz]) lap+=current[i+yz]-p;
        if (y && !solid[i-nz]) lap+=current[i-nz]-p;
        if (y+1<ny && !solid[i+nz]) lap+=current[i+nz]-p;
        if (z && !solid[i-1]) lap+=current[i-1]-p;
        if (z+1<nz && !solid[i+1]) lap+=current[i+1]-p;
        const double damping=sigma[i]*dt;
        next[i]=(2*p-(1-damping)*old[i]+r2*lap)/(1+damping);
        if (!std::isfinite(next[i])) throw std::runtime_error("nonfinite wavefield");
      }
      const double forcing=source_scale*drive[step];
      for (uint64_t a=0;a<ns;++a) {
        next[si[a]]+=forcing*sw[a]/(1+sigma[si[a]]*dt);
        if (!std::isfinite(next[si[a]])) throw std::runtime_error("nonfinite source update");
      }
      if (stride && step%stride==0) {
        double kinetic=0, potential=0;
        for (uint32_t x=0;x<nx;++x) for (uint32_t y=0;y<ny;++y) for (uint32_t z=0;z<nz;++z) {
          const uint64_t i=uint64_t(x)*yz+uint64_t(y)*nz+z;
          if (solid[i]) continue;
          const double velocity=(next[i]-current[i])/dt;
          kinetic+=velocity*velocity;
          if (x+1<nx && !solid[i+yz]) potential+=(next[i+yz]-next[i])*(current[i+yz]-current[i]);
          if (y+1<ny && !solid[i+nz]) potential+=(next[i+nz]-next[i])*(current[i+nz]-current[i]);
          if (z+1<nz && !solid[i+1]) potential+=(next[i+1]-next[i])*(current[i+1]-current[i]);
        }
        const double energy=.5*(kinetic*h*h*h+c*c*potential*h);
        if (!std::isfinite(energy)) throw std::runtime_error("nonfinite energy");
        et.push_back((double(step)+.5)*dt); en.push_back(energy);
      }
      old.swap(current); current.swap(next);
    }
    if (en.size()!=ne) throw std::runtime_error("energy count differs");
    out.write(reinterpret_cast<const char*>(et.data()),static_cast<std::streamsize>(ne*sizeof(double)));
    out.write(reinterpret_cast<const char*>(en.data()),static_cast<std::streamsize>(ne*sizeof(double)));
    if (!out) throw std::runtime_error("energy output failed");
    return 0;
  } catch (const std::exception& error) { std::cerr<<error.what()<<"\n"; return 2; }
}
