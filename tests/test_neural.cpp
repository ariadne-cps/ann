/***************************************************************************
 *            tests/test_neural.cpp
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

#include "neural/neural.hpp"
#include "ariadne-testing.hpp"

#include <string_view>

class TestNeural : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_bootstrap_version() {
        ARIADNE_TESTING_EXPECT_EQUAL(
            std::string_view(Ariadne::neural_version()),
            std::string_view("0.1.0"));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("neural");
    ARIADNE_TESTING_ADD_TEST(TestNeural, test_bootstrap_version);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
