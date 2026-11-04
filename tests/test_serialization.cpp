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
#include "neural/sha256.hpp"
#include "ariadne-testing.hpp"

#include <bit>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
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
        std::string optimizer = "test_neural_m5_optimizer.safetensors";
        ~Files() {
            (void)std::remove(json.c_str());
            (void)std::remove(weights.c_str());
            (void)std::remove(optimizer.c_str());
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

    void test_atomic_checkpoint_directory() {
        namespace fs = std::filesystem;
        Files files;
        const fs::path destination = "test_neural_m5_atomic_checkpoint";
        const fs::path snapshot_dir = "test_neural_m5_atomic_model";
        // An already published checkpoint must remain intact.
        std::error_code cleanup;
        fs::remove_all(destination, cleanup);
        fs::remove_all(snapshot_dir, cleanup);
        struct Remove {
            fs::path first, second;
            ~Remove() {
                std::error_code ec;
                fs::remove_all(first, ec);
                fs::remove_all(second, ec);
            }
        } remove{destination, snapshot_dir};
        const auto original=model();
        neural_save_snapshot_atomic(original, snapshot_dir.string());
        ARIADNE_TESTING_PROVE(fs::is_regular_file(snapshot_dir / "model.json"));
        ARIADNE_TESTING_PROVE(fs::is_regular_file(snapshot_dir / "model.safetensors"));
        ARIADNE_TESTING_PROVE(!fs::exists(snapshot_dir / "optimizer.safetensors"));
        ARIADNE_TESTING_PROVE(
            neural_load_snapshot_atomic(snapshot_dir.string()).values()
            == original.values());
        const auto preserved = file((snapshot_dir / "model.json").string());
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(original, snapshot_dir.string()));
        ARIADNE_TESTING_PROVE(
            file((snapshot_dir / "model.json").string()) == preserved);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot_atomic("test_neural_m5_missing_checkpoint"));

        NeuralOptimizerOptions options;
        options.kind=NeuralOptimizerKind::AdamW;
        NeuralTrainingSession training(original, options);
        (void)training.train_batch({{0.2, 0.4}}, {{0.6}});
        neural_save_training_checkpoint_atomic(training, destination.string());
        ARIADNE_TESTING_PROVE(fs::is_regular_file(destination / "model.json"));
        ARIADNE_TESTING_PROVE(
            fs::is_regular_file(destination / "model.safetensors"));
        ARIADNE_TESTING_PROVE(
            fs::is_regular_file(destination / "optimizer.safetensors"));
        auto restored=neural_load_training_checkpoint_atomic(destination.string());
        ARIADNE_TESTING_EXPECT_EQUAL(restored.steps(), std::size_t{1});
        ARIADNE_TESTING_PROVE(restored.export_snapshot().values() ==
                              training.export_snapshot().values());
        const auto checkpoint_bytes=file((destination / "optimizer.safetensors").string());
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint_atomic(training, destination.string()));
        ARIADNE_TESTING_PROVE(
            file((destination / "optimizer.safetensors").string()) == checkpoint_bytes);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint_atomic(training,
                "test_neural_m5_missing_parent/subdir"));

        // A malformed staging target must never be advertised as published.
        const fs::path incomplete="test_neural_m5_atomic_incomplete";
        fs::remove_all(incomplete, cleanup);
        fs::create_directory(incomplete);
        ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
            (void)neural_load_training_checkpoint_atomic(incomplete.string()));
        fs::remove_all(incomplete, cleanup);
    }

    void test_training_checkpoint_resume_and_corruption() {
        for (auto kind : {NeuralOptimizerKind::Sgd,NeuralOptimizerKind::AdamW}) {
            Files files;
            NeuralOptimizerOptions options;
            options.kind=kind;
            options.learning_rate=0.02;
            options.weight_decay=0.15;
            if (kind==NeuralOptimizerKind::Sgd) options.momentum=0.7;
            NeuralTrainingSession uninterrupted(model(),options,
                NeuralLossKind::MeanSquaredError,{false,true});
            NeuralTrainingSession paused(model(),options,
                NeuralLossKind::MeanSquaredError,{false,true});
            for (unsigned i=0u;i<3u;++i) {
                (void)uninterrupted.train_batch({{0.2,0.4}},{{0.6}});
                (void)paused.train_batch({{0.2,0.4}},{{0.6}});
            }
            neural_save_training_checkpoint(
                paused,files.json,files.weights,files.optimizer);
            auto resumed=neural_load_training_checkpoint(
                files.json,files.weights,files.optimizer);
            ARIADNE_TESTING_EXPECT_EQUAL(resumed.steps(),std::size_t{3});
            ARIADNE_TESTING_PROVE(resumed.trainable_layers() ==
                                  std::vector<bool>({false,true}));
            const auto restored_state=resumed.export_checkpoint();
            ARIADNE_TESTING_PROVE(
                restored_state.first_moment==paused.export_checkpoint().first_moment);
            ARIADNE_TESTING_PROVE(
                restored_state.second_moment==paused.export_checkpoint().second_moment);
            for (unsigned i=0u;i<5u;++i) {
                (void)uninterrupted.train_batch({{0.2,0.4}},{{0.6}});
                (void)resumed.train_batch({{0.2,0.4}},{{0.6}});
            }
            ARIADNE_TESTING_EXPECT_EQUAL(resumed.steps(),std::size_t{8});
            ARIADNE_TESTING_PROVE(
                resumed.export_snapshot().values() ==
                uninterrupted.export_snapshot().values());
            ARIADNE_TESTING_PROVE(
                resumed.export_checkpoint().first_moment ==
                uninterrupted.export_checkpoint().first_moment);
            ARIADNE_TESTING_PROVE(
                resumed.export_checkpoint().second_moment ==
                uninterrupted.export_checkpoint().second_moment);

            // Mutate one finite moment byte; topology and lengths remain valid.
            const auto original=file(files.optimizer);
            std::uint64_t header=0u;
            for (unsigned i=0u;i<8u;++i)
                header |= static_cast<std::uint64_t>(
                    static_cast<unsigned char>(original[i])) << (i*8u);
            const auto payload=8u+static_cast<std::size_t>(header);
            auto changed=original;
            changed[payload] = static_cast<char>(
                static_cast<unsigned char>(changed[payload])^1u);
            write(files.optimizer,changed);
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)neural_load_training_checkpoint(
                    files.json,files.weights,files.optimizer));
            write(files.optimizer,original);
            // Another perfectly valid model snapshot cannot be paired
            // with an optimizer state belonging to the original model.
            const auto current=paused.export_snapshot();
            auto different=current.export_values();
            different[0]+=0.125;
            neural_save_snapshot(
                NeuralParameterSnapshot(current.network(),different),
                files.json,files.weights);
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)neural_load_training_checkpoint(
                    files.json,files.weights,files.optimizer));
        }
    }

    void test_exact_binary64_roundtrip() {
        Files f;
        auto initial=model();
        neural_save_snapshot(initial,f.json,f.weights);
        const auto binary=file(f.weights);
        ARIADNE_TESTING_PROVE(binary.size()>8u);
        const auto manifest=file(f.json);
        ARIADNE_TESTING_PROVE(manifest.find("IEEE754-binary64")!=std::string::npos);
        ARIADNE_TESTING_PROVE(manifest.find("\"snapshot_sha256\"")!=std::string::npos);
        ARIADNE_TESTING_PROVE(manifest.find("\"version\"")==std::string::npos);
        ARIADNE_TESTING_EXPECT_EQUAL(neural_snapshot_sha256(initial),
                                     neural_snapshot_sha256(initial));
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

    void test_sha256_standard_vectors() {
        Detail::Sha256 empty;
        ARIADNE_TESTING_EXPECT_EQUAL(
            empty.hex_digest(),
            std::string("e3b0c44298fc1c149afbf4c8996fb924"
                        "27ae41e4649b934ca495991b7852b855"));
        Detail::Sha256 abc;
        abc.update("a");
        abc.update("bc");
        ARIADNE_TESTING_EXPECT_EQUAL(
            abc.hex_digest(),
            std::string("ba7816bf8f01cfea414140de5dae2223"
                        "b00361a396177a9cb410ff61f20015ad"));
        Detail::Sha256 multiple_blocks;
        multiple_blocks.update(std::string(1000000u, 'a'));
        ARIADNE_TESTING_EXPECT_EQUAL(
            multiple_blocks.hex_digest(),
            std::string("cdc76e5c9914fb9281a1c7e284d73e67"
                        "f1809a48a497200e046d39ccc7112cd0"));
    }

    void test_sha256_semantics() {
        auto original=model();
        auto const hash=neural_snapshot_sha256(original);
        ARIADNE_TESTING_EXPECT_EQUAL(hash.size(),std::size_t{64});
        auto bit_changed=original.export_values();
        bit_changed[1]=0.0; // -0.0 and +0.0 have distinct identities.
        ARIADNE_TESTING_PROVE(
            neural_snapshot_sha256(NeuralParameterSnapshot(original.network(),bit_changed))
                != hash);
        auto weights_changed=original.export_values();
        weights_changed[0]+=0.125;
        ARIADNE_TESTING_PROVE(
            neural_snapshot_sha256(NeuralParameterSnapshot(original.network(),weights_changed))
                != hash);
        NeuralNetwork other(2,{{2,NeuralActivation::Relu},
                               {1,NeuralActivation::Identity}});
        ARIADNE_TESTING_PROVE(
            neural_snapshot_sha256(NeuralParameterSnapshot(other,original.export_values()))
                != hash);
    }

    void test_manifest_schema_rejection() {
        Files f;
        neural_save_snapshot(model(),f.json,f.weights);
        const std::string valid=file(f.json);
        std::string altered=valid;
        // Changing topology metadata while preserving the tensor shapes must
        // not silently substitute an equivalent-sized but distinct model.
        const std::string activation_valid="\"activation\":\"tanh\"";
        const std::size_t matching=altered.find(activation_valid);
        ARIADNE_TESTING_PROVE(matching!=std::string::npos);
        altered.replace(matching,activation_valid.size(),
                        "\"activation\":\"relu\"");
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
        altered=valid;
        const std::string key="\"snapshot_sha256\":\"";
        const std::size_t digest=altered.find(key);
        ARIADNE_TESTING_PROVE(digest!=std::string::npos);
        altered[digest+key.size()]='f'==altered[digest+key.size()]?'e':'f';
        write(f.json,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        altered=valid;
        altered.replace(digest-1u, key.size()+65u, "");
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
        // Same dimensions, finite but different weight: reject by identity.
        std::uint64_t valid_header_len=0u;
        for (unsigned i=0u;i<8u;++i)
            valid_header_len |= static_cast<std::uint64_t>(
                static_cast<unsigned char>(altered[i])) << (8u*i);
        const auto payload_begin=8u+static_cast<std::size_t>(valid_header_len);
        altered[payload_begin] = static_cast<char>(
            static_cast<unsigned char>(altered[payload_begin]) ^ 1u);
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
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_atomic_checkpoint_directory);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_training_checkpoint_resume_and_corruption);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_exact_binary64_roundtrip);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_sha256_standard_vectors);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_sha256_semantics);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_manifest_schema_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_safetensors_data_rejection);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
