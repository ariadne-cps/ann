/***************************************************************************
 *            include/neural/sha256.hpp
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

#ifndef ARIADNE_NEURAL_SHA256_HPP
#define ARIADNE_NEURAL_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace Ariadne::Detail {

// FIPS 180-4 SHA-256 helper for canonical model identity.
class Sha256 {
public:
    void update(std::string_view bytes);
    void update_u64(std::uint64_t n);
    std::string hex_digest();

private:
    std::array<std::uint32_t,8u> state_ = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    std::array<std::uint8_t,64u> block_{};
    std::uint64_t bytes_processed_=0u;
    std::size_t buffered_=0u;
    void compress(std::uint8_t const* bytes);
};

} // namespace Ariadne::Detail

#endif // ARIADNE_NEURAL_SHA256_HPP
