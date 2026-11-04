/***************************************************************************
 *            tests/test_serialization.cpp
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

#include "ariadne-neural.hpp"
#include "ariadne-testing.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Ariadne;

class TestSerialization : public Testing::ContextFixture {
public:
    using ContextFixture::ContextFixture;

    struct Files {
        std::string json = "test_neural_m5_manifest.json";
        std::string weights = "test_neural_m5_weights.safetensors";
        ~Files() {
            (void)std::remove(json.c_str());
            (void)std::remove(weights.c_str());
        }
    };

    static NeuralParameterSnapshot model() {
        return NeuralParameterSnapshot(
            NeuralNetwork(2, {{2,NeuralActivation::Tanh},
                              {1,NeuralActivation::Identity}}),
            {1.5,-0.0,0.25,-2.0,0.0,-1.25,2.5,3.5,4.5});
    }

    static std::string file(std::string const& path) {
        std::ifstream input(path,std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open test fixture");
        return std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>());
    }
    static void write(std::string const& path,std::string const& bytes) {
        std::ofstream output(path,std::ios::binary|std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot write test fixture");
        output.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
        if (!output) throw std::runtime_error("Cannot write test fixture");
    }

    void test_exact_binary64_roundtrip() {
        Files f;
        auto initial=model();
        neural_save_snapshot(initial,f.json,f.weights);
        const auto binary=file(f.weights);
        ARIADNE_TESTING_PROVE(binary.size()>8u);
        const auto manifest=file(f.json);
        ARIADNE_TESTING_PROVE(manifest.find("IEEE754-binary64")!=std::string::npos);
        ARIADNE_TESTING_PROVE(binary.find("layers.0.weight")!=std::string::npos);
        auto restored=neural_load_snapshot(f.json,f.weights);
        ARIADNE_TESTING_EXPECT_EQUAL(restored.network().input_dimension(),std::size_t{2});
        ARIADNE_TESTING_EXPECT_EQUAL(restored.network().parameter_count(),std::size_t{9});
        ARIADNE_TESTING_EXPECT_EQUAL(restored.network().layers().size(),std::size_t{2});
        for (std::size_t i=0u;i<initial.values().size();++i) {
            ARIADNE_TESTING_EXPECT_EQUAL(
                std::bit_cast<std::uint64_t>(initial.values().at(i)),
                std::bit_cast<std::uint64_t>(restored.values().at(i)));
        }
        ARIADNE_TESTING_PROVE(std::signbit(restored.values().at(1)));
        NeuralExecutionSession execution(restored);
        ARIADNE_TESTING_PROVE(std::isfinite(execution.infer({0.5,0.25}).at(0)));
        // Saving an unmodified parsed model is byte-for-byte deterministic.
        neural_save_snapshot(restored,f.json,f.weights);
        ARIADNE_TESTING_PROVE(file(f.weights)==binary);
        ARIADNE_TESTING_PROVE(file(f.json)==manifest);
    }

    void test_manifest_schema_rejection() {
        Files f;
        neural_save_snapshot(model(),f.json,f.weights);
        const std::string valid=file(f.json);
        const std::string version="\"version\":1";
        const std::size_t pos=valid.find(version);
        ARIADNE_TESTING_PROVE(pos!=std::string::npos);
        std::string altered=valid;
        altered.replace(pos,version.size(),"\"version\":2");
        write(f.json,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        altered=valid;
        const std::string activation="\"activation\":\"tanh\"";
        const std::size_t act=altered.find(activation);
        ARIADNE_TESTING_PROVE(act!=std::string::npos);
        altered.replace(act,activation.size(),"\"activation\":\"unknown\"");
        write(f.json,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        altered=valid;
        const std::string marker="\"tensors\":\"safetensors\"";
        const std::size_t end=altered.find(marker);
        ARIADNE_TESTING_PROVE(end!=std::string::npos);
        altered.replace(end,marker.size(),
            "\"tensors\":\"safetensors\",\"unknown\":0");
        write(f.json,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        write(f.json,valid);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(model(),f.json,f.json));
    }

    void test_safetensors_data_rejection() {
        Files f;
        neural_save_snapshot(model(),f.json,f.weights);
        const std::string original=file(f.weights);
        write(f.weights,original.substr(0u,original.size()-1u));
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        std::string altered=original;
        altered.push_back('\0');
        write(f.weights,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        altered=original;
        const std::size_t dtype=altered.find("\"F64\"");
        ARIADNE_TESTING_PROVE(dtype!=std::string::npos);
        altered.replace(dtype,5u,"\"F32\"");
        write(f.weights,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        altered=original;
        // Byte-for-byte mutation to a NaN in the first payload double.
        std::uint64_t header_len=0u;
        for (unsigned i=0u;i<8u;++i)
            header_len |= static_cast<std::uint64_t>(
                static_cast<unsigned char>(altered[i])) << (i*8u);
        const std::size_t offset=8u+static_cast<std::size_t>(header_len);
        const std::uint64_t qnan=UINT64_C(0x7ff8000000000000);
        for (unsigned i=0u;i<8u;++i)
            altered[offset+i]=static_cast<char>((qnan>>(i*8u))&0xffu);
        write(f.weights,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
    }
};

int main(int argc,char** argv) {
    ARIADNE_TESTING_SUITE("serialization");
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_exact_binary64_roundtrip);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_manifest_schema_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_safetensors_data_rejection);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
