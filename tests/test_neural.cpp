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
