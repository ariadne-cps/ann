/***************************************************************************
 *            neural/neural.hpp
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

#ifndef ARIADNE_NEURAL_HPP
#define ARIADNE_NEURAL_HPP

namespace Ariadne {

/** @file neural/neural.hpp
 *  @brief Ariadne Neural module version and bootstrap declarations.
 */

/** @brief Return the Ariadne Neural module version.
 *  @return A static, null-terminated semantic-version string.
 */
const char* neural_version() noexcept;

} // namespace Ariadne

#endif // ARIADNE_NEURAL_HPP
