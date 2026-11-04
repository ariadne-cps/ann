/***************************************************************************
 *            src/sha256.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neural.
 *
 *  Ariadne Neural is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neural is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neural.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "neural/sha256.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace Ariadne::Detail {
namespace {

constexpr std::array<std::uint32_t,64u> constants = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

} // namespace

void Sha256::update(std::string_view bytes) {
        bytes_processed_ += static_cast<std::uint64_t>(bytes.size());
        if (buffered_ != 0u) {
            const auto count = std::min(bytes.size(), block_.size()-buffered_);
            std::memcpy(block_.data()+buffered_, bytes.data(), count);
            buffered_ += count;
            bytes.remove_prefix(count);
            if (buffered_ == block_.size()) {
                compress(block_.data());
                buffered_ = 0u;
            }
        }
        while (bytes.size() >= block_.size()) {
            compress(reinterpret_cast<std::uint8_t const*>(bytes.data()));
            bytes.remove_prefix(block_.size());
        }
        if (!bytes.empty()) {
            std::memcpy(block_.data(),bytes.data(),bytes.size());
            buffered_ = bytes.size();
        }
    }

void Sha256::update_u64(std::uint64_t n) {
        std::array<char,8u> bytes{};
        for (unsigned i=0u;i<8u;++i)
            bytes[i]=static_cast<char>((n>>(8u*i))&0xffu);
        update(std::string_view(bytes.data(),bytes.size()));
    }

std::string Sha256::hex_digest() {
        const auto bits=bytes_processed_*8u;
        std::array<char,64u> padding{};
        padding[0]=static_cast<char>(0x80);
        const std::size_t fill=buffered_<56u?56u-buffered_:120u-buffered_;
        update(std::string_view(padding.data(),fill));
        std::array<char,8u> length{};
        for (unsigned i=0u;i<8u;++i)
            length[i]=static_cast<char>((bits>>(8u*(7u-i)))&0xffu);
        update(std::string_view(length.data(),length.size()));
        constexpr char digits[]="0123456789abcdef";
        std::string result(64u,'0');
        for (std::size_t i=0u;i<state_.size();++i)
            for (unsigned j=0u;j<8u;++j)
                result[8u*i+j]=digits[(state_[i]>>(4u*(7u-j)))&0xfu];
        return result;
    }

void Sha256::compress(std::uint8_t const* bytes) {
        std::array<std::uint32_t,64u> w{};
        for (std::size_t i=0u;i<16u;++i)
            w[i]=(static_cast<std::uint32_t>(bytes[4u*i])<<24u)
                |(static_cast<std::uint32_t>(bytes[4u*i+1u])<<16u)
                |(static_cast<std::uint32_t>(bytes[4u*i+2u])<<8u)
                |static_cast<std::uint32_t>(bytes[4u*i+3u]);
        for (std::size_t i=16u;i<64u;++i) {
            const auto a=std::rotr(w[i-15u],7)^std::rotr(w[i-15u],18)^(w[i-15u]>>3u);
            const auto b=std::rotr(w[i-2u],17)^std::rotr(w[i-2u],19)^(w[i-2u]>>10u);
            w[i]=w[i-16u]+a+w[i-7u]+b;
        }
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        auto e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (std::size_t i=0u;i<64u;++i) {
            const auto sigma1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);
            const auto choose=(e&f)^(~e&g);
            const auto t1=h+sigma1+choose+constants[i]+w[i];
            const auto sigma0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);
            const auto majority=(a&b)^(a&c)^(b&c);
            const auto t2=sigma0+majority;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }

} // namespace Ariadne::Detail
