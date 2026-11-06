/***************************************************************************
 *            tests/test_serialization.cpp
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

#include "ariadne-ann.hpp"
#include "ann/sha256.hpp"
#include "ariadne-testing.hpp"

#include <bit>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <csignal>
#include <sys/resource.h>
#endif

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
        // Every activation contributes its own code to the digest.
        NeuralNetwork all(1, {{1,NeuralActivation::Identity},
                              {1,NeuralActivation::Relu},
                              {1,NeuralActivation::Tanh},
                              {1,NeuralActivation::Sigmoid},
                              {1,NeuralActivation::Softplus}});
        const auto all_hash = neural_snapshot_sha256(NeuralParameterSnapshot(
            all, std::vector<double>(all.parameter_count(), 0.5)));
        ARIADNE_TESTING_EXPECT_EQUAL(all_hash.size(), std::size_t{64});
        NeuralNetwork sigmoid_only(1, {{1,NeuralActivation::Sigmoid}});
        ARIADNE_TESTING_PROVE(
            neural_snapshot_sha256(NeuralParameterSnapshot(
                sigmoid_only, std::vector<double>(2u, 0.5))) != all_hash);
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
        altered=valid;
        // A non-hex character in the digest.
        altered[digest+key.size()]='g';
        write(f.json,altered);
        ARIADNE_TESTING_EXPECT_THROWS(
            std::invalid_argument,(void)neural_load_snapshot(f.json,f.weights));
        write(f.json,valid);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(model(),f.json,f.json));
    }

    std::uint64_t safetensors_header_length(std::string const& binary) {
        std::uint64_t n = 0u;
        for (unsigned i = 0u; i < 8u; ++i)
            n |= static_cast<std::uint64_t>(
                static_cast<unsigned char>(binary[i])) << (8u*i);
        return n;
    }
    std::string safetensors_header(std::string const& binary) {
        return binary.substr(8u, safetensors_header_length(binary));
    }
    // Rebuild a safetensors file with a replacement header, keeping payload.
    std::string safetensors_with_header(std::string const& binary,
                                        std::string const& header) {
        std::string out;
        const std::uint64_t n = header.size();
        for (unsigned i = 0u; i < 8u; ++i)
            out.push_back(static_cast<char>((n >> (8u*i)) & 0xffu));
        out += header;
        out += binary.substr(8u + safetensors_header_length(binary));
        return out;
    }
    std::string replace_once(std::string s, std::string const& from,
                             std::string const& to) {
        const auto pos = s.find(from);
        ARIADNE_TESTING_PROVE(pos != std::string::npos);
        s.replace(pos, from.size(), to);
        return s;
    }

    void test_json_and_manifest_rejection() {
        Files f;
        neural_save_snapshot(model(), f.json, f.weights);
        const std::string valid = file(f.json);
        auto expect_reject = [&](std::string const& manifest) {
            write(f.json, manifest);
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)neural_load_snapshot(f.json, f.weights));
        };
        // Trailing content after a valid document.
        expect_reject(valid + "x");
        // Space, tab, CR and LF are tolerated between tokens.
        write(f.json, "\t\r\n " + valid);
        ARIADNE_TESTING_PROVE(
            neural_load_snapshot(f.json, f.weights).values() == model().values());
        // Unterminated string.
        expect_reject(replace_once(valid, "\"tensors\":\"safetensors\"",
                                   "\"tensors\":\"safetensors"));
        // Forbidden characters in strings: backslash, control byte, DEL.
        expect_reject(replace_once(valid, "\"tensors\":\"safetensors\"",
                                   "\"tensors\":\"safe\\tensors\""));
        expect_reject(replace_once(valid, "\"tensors\":\"safetensors\"",
                                   "\"tensors\":\"safe\x01tensors\""));
        expect_reject(replace_once(valid, "\"tensors\":\"safetensors\"",
                                   "\"tensors\":\"safe\x7ftensors\""));
        // Number expected (below '0', above '9'), noncanonical leading zero
        // (also with the digit at the very end of the input), uint64 overflow.
        expect_reject("{\"a\":0");
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":x"));
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":!"));
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":02"));
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":0x"));
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":99999999999999999999999999"));
        // Empty document, nesting too deep, oversized array.
        expect_reject("");
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":" + std::string(20u, '[')
                                   + std::string(20u, ']')));
        {
            std::string big_array = "\"input_dimension\":[";
            for (unsigned i = 0u; i < 16385u; ++i) {
                big_array += "0";
                big_array += ",";
            }
            big_array += "0]";
            expect_reject(replace_once(valid, "\"input_dimension\":2", big_array));
        }
        // Duplicate key and oversized object.
        expect_reject(replace_once(valid, "\"format\":\"ariadne-neural\"",
                                   "\"format\":\"ariadne-neural\",\"format\":\"ariadne-neural\""));
        {
            std::string big_object = "{";
            for (unsigned i = 0u; i < 16385u; ++i) {
                big_object += "\"k" + std::to_string(i) + "\":0,";
            }
            big_object += "\"k\":0}";
            expect_reject(big_object);
        }
        // Missing field (renamed key keeps the field count), wrong types,
        // unsupported literal values.
        expect_reject(replace_once(valid, "\"numeric\":\"IEEE754-binary64\"",
                                   "\"numerical\":\"IEEE754-binary64\""));
        expect_reject(replace_once(valid, "\"input_dimension\":2",
                                   "\"input_dimension\":\"2\""));
        expect_reject(replace_once(valid, "\"format\":\"ariadne-neural\"",
                                   "\"format\":0"));
        expect_reject(replace_once(valid, "\"format\":\"ariadne-neural\"",
                                   "\"format\":\"other\""));
        // Invalid layer collections and layer entries.
        expect_reject(replace_once(valid, "\"layers\":[{\"outputs\":2,\"activation\":\"tanh\"},{\"outputs\":1,\"activation\":\"identity\"}]",
                                   "\"layers\":{}"));
        expect_reject(replace_once(valid, "\"layers\":[{\"outputs\":2,\"activation\":\"tanh\"},{\"outputs\":1,\"activation\":\"identity\"}]",
                                   "\"layers\":[]"));
        {
            std::string many = "\"layers\":[";
            for (unsigned i = 0u; i < 4097u; ++i)
                many += "{\"outputs\":1,\"activation\":\"identity\"},";
            many += "{\"outputs\":1,\"activation\":\"identity\"}]";
            expect_reject(replace_once(valid, "\"layers\":[{\"outputs\":2,\"activation\":\"tanh\"},{\"outputs\":1,\"activation\":\"identity\"}]", many));
        }
        expect_reject(replace_once(valid, "{\"outputs\":2,\"activation\":\"tanh\"}",
                                   "{\"outputs\":2,\"activation\":\"tanh\",\"extra\":0}"));
        // Layer entries must be objects.
        expect_reject(replace_once(valid, "\"layers\":[{\"outputs\":2,\"activation\":\"tanh\"},{\"outputs\":1,\"activation\":\"identity\"}]",
                                   "\"layers\":[0,0]"));
        // Topology whose parameter count exceeds the resource limit.
        expect_reject("{\"format\":\"ariadne-neural\",\"numeric\":\"IEEE754-binary64\","
                      "\"input_dimension\":5793,\"layers\":[{\"outputs\":5793,"
                      "\"activation\":\"identity\"}],\"tensors\":\"safetensors\","
                      "\"snapshot_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"}");
        write(f.json, valid);
    }

    void test_safetensors_header_rejection() {
        Files f;
        neural_save_snapshot(model(), f.json, f.weights);
        const std::string binary = file(f.weights);
        const std::string header = safetensors_header(binary);
        auto expect_reject = [&](std::string const& bytes) {
            write(f.weights, bytes);
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)neural_load_snapshot(f.json, f.weights));
        };
        auto expect_header_reject = [&](std::string const& new_header) {
            expect_reject(safetensors_with_header(binary, new_header));
        };
        // Truncated file and invalid header-length prefixes.
        expect_reject(binary.substr(0u, 4u));
        {
            std::string b(8u, '\0');
            b += binary.substr(8u);
            expect_reject(b);  // header length zero
        }
        {
            std::string b(8u, '\xff');
            b += binary.substr(8u);
            expect_reject(b);  // header length beyond the limit
        }
        {
            std::string b = binary;
            const std::uint64_t huge = 1000u;
            for (unsigned i = 0u; i < 8u; ++i)
                b[i] = static_cast<char>((huge >> (8u*i)) & 0xffu);
            expect_reject(b);  // header length beyond the file size
        }
        // Header is not JSON, not an object, or has the wrong tensor count.
        expect_header_reject("xxx");
        expect_header_reject("[]");
        expect_header_reject("{}");
        // A tensor is missing under its canonical name.
        expect_header_reject(replace_once(header, "\"layers.1.bias\"",
                                          "\"layers.1.biasX\""));
        // Invalid shape/offsets structures.
        expect_header_reject(replace_once(header, "\"shape\":[2,2]", "\"shape\":0"));
        expect_header_reject(replace_once(header, "\"data_offsets\":[0,32]",
                                          "\"data_offsets\":0"));
        expect_header_reject(replace_once(header, "\"shape\":[2,2]",
                                          "\"shape\":[2,2,2]"));
        expect_header_reject(replace_once(header, "\"shape\":[2,2]",
                                          "\"shape\":[0,2]"));
        expect_header_reject(replace_once(header, "\"shape\":[2,2]",
                                          "\"shape\":[40000000,2]"));
        expect_header_reject(replace_once(header, "\"data_offsets\":[0,32]",
                                          "\"data_offsets\":[0]"));
        expect_header_reject(replace_once(header, "\"data_offsets\":[0,32]",
                                          "\"data_offsets\":[32,0]"));
        expect_header_reject(replace_once(header, "\"data_offsets\":[0,32]",
                                          "\"data_offsets\":[0,8]"));
        // Self-consistent tensor with the wrong shape for its slot.
        expect_header_reject(replace_once(header, "\"shape\":[2,2],\"data_offsets\":[0,32]",
                                          "\"shape\":[1,4],\"data_offsets\":[0,32]"));
        write(f.weights, binary);
    }

    void test_training_checkpoint_file_rejection() {
        Files f;
        NeuralOptimizerOptions opt;
        opt.kind = NeuralOptimizerKind::Sgd;
        opt.learning_rate = 0.02;
        opt.momentum = 0.7;
        NeuralTrainingSession trainer(model(), opt);
        (void)trainer.train_batch({{0.2, 0.4}}, {{0.6}});
        neural_save_training_checkpoint(trainer, f.json, f.weights, f.optimizer);
        const std::string opt_bin = file(f.optimizer);
        const std::string opt_header = safetensors_header(opt_bin);
        auto expect_reject = [&](std::string const& optimizer_bytes) {
            write(f.optimizer, optimizer_bytes);
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                (void)neural_load_training_checkpoint(f.json, f.weights, f.optimizer));
        };
        auto expect_header_reject = [&](std::string const& new_header) {
            expect_reject(safetensors_with_header(opt_bin, new_header));
        };
        // Distinct-path validation on save and load.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, "", f.weights, f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, f.json, "", f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, f.json, f.weights, ""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, f.json, f.json, f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, f.json, f.weights, f.weights));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_training_checkpoint(trainer, f.json, f.weights, f.json));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint(f.json, f.json, f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint("", f.weights, f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint(f.json, "", f.optimizer));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint(f.json, f.weights, ""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint(f.json, f.weights, f.json));
        // Truncated optimizer file and invalid header-length prefixes.
        expect_reject(opt_bin.substr(0u, 4u));
        {
            std::string b(8u, '\0');
            b += opt_bin.substr(8u);
            expect_reject(b);
        }
        {
            std::string b(8u, '\xff');
            b += opt_bin.substr(8u);
            expect_reject(b);
        }
        {
            std::string b = opt_bin;
            const std::uint64_t beyond = opt_bin.size();  // > size-8
            for (unsigned i = 0u; i < 8u; ++i)
                b[i] = static_cast<char>((beyond >> (8u*i)) & 0xffu);
            expect_reject(b);
        }
        // Wrong top-level key count, metadata key count, literal format.
        expect_header_reject(replace_once(
            opt_header,
            ",\"second_moment\":{\"dtype\":\"F64\",\"shape\":[9],\"data_offsets\":[72,144]}",
            ""));
        expect_header_reject(replace_once(opt_header, "\"loss\":\"mse\",", ""));
        expect_header_reject(replace_once(opt_header,
            "\"format\":\"ariadne-neural-training\"", "\"format\":\"other\""));
        // Missing metadata field under its canonical name.
        expect_header_reject(replace_once(opt_header, "\"__metadata__\"",
                                          "\"__meta__\""));
        // Step count encodings: leading zero, empty, non-digit, overflow.
        expect_header_reject(replace_once(opt_header, "\"steps\":\"1\"",
                                          "\"steps\":\"01\""));
        expect_header_reject(replace_once(opt_header, "\"steps\":\"1\"",
                                          "\"steps\":\"\""));
        expect_header_reject(replace_once(opt_header, "\"steps\":\"1\"",
                                          "\"steps\":\"1x\""));
        expect_header_reject(replace_once(opt_header, "\"steps\":\"1\"",
                                          "\"steps\":\"-1\""));
        expect_header_reject(replace_once(opt_header, "\"steps\":\"1\"",
                                          "\"steps\":\"999999999999999999999999999999\""));
        // Trainable mask: wrong length, invalid character.
        expect_header_reject(replace_once(opt_header, "\"trainable_layers\":\"11\"",
                                          "\"trainable_layers\":\"1\""));
        expect_header_reject(replace_once(opt_header, "\"trainable_layers\":\"11\"",
                                          "\"trainable_layers\":\"1x\""));
        // Optimizer kind and loss names.
        expect_header_reject(replace_once(opt_header, "\"optimizer\":\"sgd\"",
                                          "\"optimizer\":\"rmsprop\""));
        expect_header_reject(replace_once(opt_header, "\"loss\":\"mse\"",
                                          "\"loss\":\"huber\""));
        // Numeric encodings: wrong hex length, invalid hex digit.
        expect_header_reject(replace_once(opt_header, "\"learning_rate\":\"",
                                          "\"learning_rate\":\"0"));
        for (char bad : {'g', ':', '!'}) {
            std::string h = opt_header;
            const auto pos = h.find("\"momentum\":\"");
            ARIADNE_TESTING_PROVE(pos != std::string::npos);
            const std::size_t start =
                pos + std::string("\"momentum\":\"").size();
            h.replace(start, 16u, std::string(16u, bad));
            expect_header_reject(h);
        }
        // Moment tensor missing or with the wrong offset order.
        expect_header_reject(replace_once(opt_header, "\"first_moment\"",
                                          "\"first_momentX\""));
        expect_header_reject(replace_once(opt_header, "\"data_offsets\":[0,72]",
                                          "\"data_offsets\":[8,80]"));
        // Truncated payload: offsets no longer match the file size.
        expect_reject(opt_bin.substr(0u, opt_bin.size() - 8u));
        // Corrupted checkpoint digest: wrong length, then wrong hex content.
        {
            std::string h = opt_header;
            const auto pos = h.find("\"checkpoint_sha256\":\"");
            ARIADNE_TESTING_PROVE(pos != std::string::npos);
            const std::size_t start =
                pos + std::string("\"checkpoint_sha256\":\"").size();
            h.erase(start + 63u, 1u);  // 63 hex chars
            expect_header_reject(h);
            std::string h2 = opt_header;
            const std::size_t start2 =
                h2.find("\"checkpoint_sha256\":\"") +
                std::string("\"checkpoint_sha256\":\"").size();
            h2[start2] = (h2[start2] == '0') ? '1' : '0';  // still 64 hex chars
            expect_header_reject(h2);
            std::string h3 = opt_header;
            const std::size_t start3 =
                h3.find("\"checkpoint_sha256\":\"") +
                std::string("\"checkpoint_sha256\":\"").size();
            h3[start3] = 'g';  // non-hex character
            expect_header_reject(h3);
        }
        // Optimizer path equal to the weights path on load.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint(f.json, f.weights, f.weights));
        write(f.optimizer, opt_bin);
    }

    void test_io_failure_paths() {
        Files f;
        neural_save_snapshot(model(), f.json, f.weights);
        // Missing input file.
        ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
            (void)neural_load_snapshot("test_neural_m5_missing.json", f.weights));
        // A directory cannot be read as a file.
        {
            namespace fs = std::filesystem;
            const fs::path dir = "test_neural_m5_dir_input";
            std::error_code ec;
            fs::remove_all(dir, ec);
            fs::create_directory(dir);
            ARIADNE_TESTING_EXPECT_EXCEPTION(
                (void)neural_load_snapshot(dir.string(), f.weights));
            fs::remove_all(dir, ec);
        }
        // Manifest beyond the size limit.
        write(f.json, std::string(1024u*1024u + 1u, ' '));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot(f.json, f.weights));
        // Output into a missing directory.
        ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
            neural_save_snapshot(model(), "test_neural_m5_missing_dir/m.json",
                                 f.weights));
#if !defined(_WIN32)
        // A file size limit makes the write fail after a successful open.
        // POSIX only: Windows has no equivalent mechanism, so this
        // write-failure branch is exercised on POSIX platforms.
        {
            struct rlimit original;
            struct rlimit small;
            (void)getrlimit(RLIMIT_FSIZE, &original);
            small.rlim_cur = 64u;
            small.rlim_max = original.rlim_max;
            (void)setrlimit(RLIMIT_FSIZE, &small);
            (void)std::signal(SIGXFSZ, SIG_IGN);
            ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
                neural_save_snapshot(model(), f.json, f.weights));
            (void)setrlimit(RLIMIT_FSIZE, &original);
            (void)std::signal(SIGXFSZ, SIG_DFL);
        }
#endif
        // Snapshot path validation.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(model(), "", f.weights));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(model(), f.json, ""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(model(), f.json, f.json));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot("", f.weights));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot(f.json, ""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot(f.json, f.json));
    }

    void test_bce_checkpoint_roundtrip() {
        Files f;
        NeuralOptimizerOptions opt;
        opt.kind = NeuralOptimizerKind::AdamW;
        opt.learning_rate = 0.05;
        NeuralTrainingSession trainer(model(), opt,
            NeuralLossKind::BinaryCrossEntropyWithLogits);
        (void)trainer.train_batch({{0.2, 0.4}}, {{1.0}});
        neural_save_training_checkpoint(trainer, f.json, f.weights, f.optimizer);
        auto resumed = neural_load_training_checkpoint(f.json, f.weights, f.optimizer);
        ARIADNE_TESTING_EXPECT_EQUAL(resumed.steps(), std::size_t{1});
        ARIADNE_TESTING_PROVE(resumed.export_checkpoint().loss ==
                              NeuralLossKind::BinaryCrossEntropyWithLogits);
        ARIADNE_TESTING_PROVE(resumed.export_snapshot().values() ==
                              trainer.export_snapshot().values());
        (void)resumed.train_batch({{0.2, 0.4}}, {{0.0}});
        ARIADNE_TESTING_EXPECT_EQUAL(resumed.steps(), std::size_t{2});
    }

    void test_atomic_checkpoint_rejection() {
        namespace fs = std::filesystem;
        const auto snap = model();
        std::error_code ec;
        // Empty directory names.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(snap, ""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_snapshot_atomic(""));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            (void)neural_load_training_checkpoint_atomic(""));
        // Names that do not designate a new directory.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(snap, "."));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(snap, ".."));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(snap, "/"));
        // A dangling symlink destination is rejected (exists() follows links).
        const fs::path link = "test_neural_m5_dangling_link";
        fs::remove(link, ec);
        fs::create_symlink("test_neural_m5_nonexistent_target", link, ec);
        ARIADNE_TESTING_PROVE(!ec);
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(snap, link.string()));
        fs::remove(link, ec);
        // An overlong destination component makes every staging slot fail,
        // so allocation is refused after all attempts.
        ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
            neural_save_snapshot_atomic(snap, std::string(300u, 'a')));
    }

    void test_atomic_checkpoint_staging() {
        namespace fs = std::filesystem;
        const auto snap = model();
        const fs::path parent = "test_neural_m5_staging_parent";
        std::error_code ec;
        fs::remove_all(parent, ec);
        fs::create_directory(parent);
        struct Clean {
            fs::path p;
            ~Clean() { std::error_code e; fs::remove_all(p, e); }
        } clean{parent};

        // A pre-existing file in the first staging slot forces the next one.
        { std::ofstream blocker((parent / "ckpt.staging.0").string()); }
        neural_save_snapshot_atomic(snap, (parent / "ckpt").string());
        ARIADNE_TESTING_PROVE(fs::is_regular_file(parent / "ckpt" / "model.json"));
        ARIADNE_TESTING_PROVE(
            neural_load_snapshot_atomic((parent / "ckpt").string()).values()
            == snap.values());
        ARIADNE_TESTING_PROVE(fs::is_regular_file(parent / "ckpt.staging.0"));
        for (auto const& e : fs::directory_iterator(parent))
            if (e.path().filename().string().rfind("ckpt.staging.", 0) == 0)
                ARIADNE_TESTING_EXPECT_EQUAL(e.path().filename().string(),
                                            std::string("ckpt.staging.0"));

        // All 64 staging slots occupied: allocation fails, nothing publishes.
        const fs::path blocked = parent / "blocked";
        for (unsigned i = 0u; i < 64u; ++i) {
            std::ofstream b(
                (parent / ("blocked.staging." + std::to_string(i))).string());
        }
        ARIADNE_TESTING_EXPECT_THROWS(std::runtime_error,
            neural_save_snapshot_atomic(snap, blocked.string()));
        ARIADNE_TESTING_PROVE(!fs::exists(blocked));

        // A model beyond the resource limit fails only after staging starts,
        // so the staging directory must be cleaned up again.
        std::vector<NeuralDenseLayerSpec> layers(4097u, {1, NeuralActivation::Identity});
        NeuralParameterSnapshot huge(NeuralNetwork(1, std::move(layers)),
                                     std::vector<double>(4097u * 2u, 0.5));
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot_atomic(huge, (parent / "huge").string()));
        ARIADNE_TESTING_PROVE(!fs::exists(parent / "huge"));
        for (auto const& e : fs::directory_iterator(parent))
            ARIADNE_TESTING_PROVE(
                e.path().filename().string().rfind("huge.staging.", 0) != 0);
        // A direct save of the oversized network hits the same limit.
        ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
            neural_save_snapshot(huge, "test_neural_m5_huge.json",
                                 "test_neural_m5_huge.safetensors"));
    }

    void test_atomic_checkpoint_publish_race() {
        namespace fs = std::filesystem;
        // Large enough that writing and re-reading the staged copy takes far
        // longer than the helper thread needs to notice the staging
        // directory and occupy the destination (~2 MB payload).
        const std::size_t dim = 512u;
        const auto big = NeuralParameterSnapshot(
            NeuralNetwork(dim, {{dim, NeuralActivation::Identity}}),
            std::vector<double>(dim * dim + dim, 0.25));
        const fs::path parent = "test_neural_m5_race_parent";
        std::error_code ec;
        fs::remove_all(parent, ec);
        fs::create_directory(parent);
        struct Clean {
            fs::path p;
            ~Clean() { std::error_code e; fs::remove_all(p, e); }
        } clean{parent};

        auto race = [&](bool use_symlink) {
            const fs::path dest =
                parent / (use_symlink ? "raced_link" : "raced_dir");
            std::atomic<bool> stop{false};
            std::atomic<bool> occupied{false};
            std::thread helper([&] {
                const std::string prefix =
                    dest.filename().string() + ".staging.";
                while (!stop.load(std::memory_order_relaxed)) {
                    for (auto const& e : fs::directory_iterator(parent)) {
                        if (e.path().filename().string().rfind(prefix, 0) == 0) {
                            std::error_code sec;
                            if (use_symlink)
                                fs::create_symlink("test_neural_m5_missing_target",
                                                   dest, sec);
                            else
                                fs::create_directory(dest, sec);
                            occupied.store(!sec);
                            stop.store(true);
                            return;
                        }
                    }
                    std::this_thread::yield();
                }
            });
            ARIADNE_TESTING_EXPECT_THROWS(std::invalid_argument,
                neural_save_snapshot_atomic(big, dest.string()));
            stop.store(true);
            helper.join();
            // The destination appeared after the entry check but before the
            // publish rename, so the atomic save refused to replace it.
            ARIADNE_TESTING_PROVE(occupied.load());
            ARIADNE_TESTING_PROVE(fs::exists(dest) || fs::is_symlink(dest));
            ARIADNE_TESTING_PROVE(!fs::exists(dest / "model.json"));
        };
        race(false);  // existing directory at publish time
        race(true);   // dangling symlink at publish time
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
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_json_and_manifest_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_safetensors_header_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_training_checkpoint_file_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_io_failure_paths);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_bce_checkpoint_roundtrip);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_atomic_checkpoint_rejection);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_atomic_checkpoint_staging);
    ARIADNE_TESTING_ADD_TEST(TestSerialization,test_atomic_checkpoint_publish_race);
    return ARIADNE_TESTING_RUN_TESTS_WITH_ARGS(argc,argv);
}
