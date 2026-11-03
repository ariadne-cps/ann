// The umbrella header must be self-contained when included first.
#include "ariadne-neural.hpp"
#include "ariadne-testing.hpp"

#include <string_view>

class TestNeuralUmbrella : public Ariadne::Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    void test_public_header() {
        ARIADNE_TESTING_EXPECT_EQUAL(
            std::string_view(Ariadne::neural_version()),
            std::string_view("0.1.0"));
    }
};

int main(int argc, char** argv) {
    ARIADNE_TESTING_SUITE("neural umbrella");
    ARIADNE_TESTING_ADD_TEST(TestNeuralUmbrella, test_public_header);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc, argv);
}
