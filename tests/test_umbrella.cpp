/***************************************************************************
 *            tests/test_umbrella.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne ANN.
 *
 *  Ariadne ANN is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne ANN is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne ANN.  If not, see <https://www.gnu.org/licenses/>.
 */

// The umbrella header must be self-contained when included first.
#include "ariadne-ann.hpp"
#include "ariadne-testing.hpp"

#include <string_view>

class TestANNUmbrella : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_public_header() {
        ARIADNE_TESTING_EXPECT_EQUAL(
            std::string_view(Ariadne::ann_version()),
            std::string_view("0.1.0"));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("ann umbrella");
    ARIADNE_TESTING_ADD_TEST(TestANNUmbrella, test_public_header);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
