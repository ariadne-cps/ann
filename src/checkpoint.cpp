/***************************************************************************
 *            src/checkpoint.cpp
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
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne ANN. If not, see <https://www.gnu.org/licenses/>.
 */

/** @file src/checkpoint.cpp
 *  @brief Implement atomic checkpoint directory publication.
 */

#include "ann/serialization.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace Ariadne {
namespace {

namespace fs = std::filesystem;

constexpr char model_manifest[] = "model.json";
constexpr char model_weights[] = "model.safetensors";
constexpr char optimizer_weights[] = "optimizer.safetensors";

// The parent must already exist. Never create the destination directory
// before all its files have been written and checked.
fs::path publication_target(std::string const& directory) {
    if (directory.empty())
        throw std::invalid_argument("Checkpoint directory must be nonempty");
    const fs::path target(directory);
    if (target.filename().empty() || target.filename() == "." ||
        target.filename() == "..")
        throw std::invalid_argument("Checkpoint must name a new directory");
    const fs::path parent = target.has_parent_path() ? target.parent_path() : fs::path(".");
    if (!fs::is_directory(parent))
        throw std::invalid_argument("Checkpoint parent directory does not exist");
    // error_code overloads: an unstatable name is simply treated as absent
    // and is rejected later, when staging allocation fails.
    std::error_code ec;
    if (fs::exists(target, ec) || fs::is_symlink(target, ec))
        throw std::invalid_argument("Checkpoint destination already exists");
    return target;
}

fs::path existing_checkpoint(std::string const& directory) {
    if (directory.empty())
        throw std::invalid_argument("Checkpoint directory must be nonempty");
    const fs::path path(directory);
    if (!fs::is_directory(path))
        throw std::invalid_argument("Checkpoint directory does not exist");
    return path;
}

class StagingDirectory {
public:
    explicit StagingDirectory(fs::path const& destination) {
        const fs::path parent = destination.has_parent_path()
            ? destination.parent_path() : fs::path(".");
        // Candidate names are deterministic (destination.staging.N): the
        // existence check plus the attempt suffix handle collisions, so no
        // clock or process-local entropy is needed.
        const std::string prefix = destination.filename().string()
            + ".staging.";
        std::error_code ec;
        for (unsigned attempt = 0u; attempt != 64u; ++attempt) {
            const auto candidate = parent / (prefix + std::to_string(attempt));
            if (fs::exists(candidate, ec))
                continue;
            ec.clear();
            if (fs::create_directory(candidate, ec)) {
                path_ = candidate;
                return;
            }
            // Creation failed (permissions, name too long, ...): the next
            // slot is tried; persistent failure is reported after the loop.
        }
        throw std::runtime_error(
            "Cannot allocate checkpoint staging directory: " + ec.message());
    }

    StagingDirectory(StagingDirectory const&) = delete;
    StagingDirectory& operator=(StagingDirectory const&) = delete;

    ~StagingDirectory() {
        if (!path_.empty()) {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
    }

    fs::path const& path() const noexcept { return path_; }

    void publish(fs::path const& destination) {
        // Do not intentionally replace existing checkpoints, including a
        // dangling symlink or a preexisting empty destination directory.
        std::error_code ec;
        if (fs::exists(destination, ec) || fs::is_symlink(destination, ec))
            throw std::invalid_argument("Checkpoint destination already exists");
        // The staging directory is a sibling of the destination. This single
        // rename makes all files visible together on a supported filesystem.
        fs::rename(path_, destination);
        path_.clear();  // Ownership transferred; cleanup must not touch it.
    }

private:
    fs::path path_;
};

fs::path file(fs::path const& directory, char const* basename) {
    return directory / basename;
}

} // namespace

void neural_save_snapshot_atomic(
    NeuralParameterSnapshot const& snapshot,
    std::string const& directory) {
    const auto destination = publication_target(directory);
    StagingDirectory stage(destination);
    neural_save_snapshot(snapshot,
                         file(stage.path(), model_manifest).string(),
                         file(stage.path(), model_weights).string());
    // Verify the exact persisted representation before advertising it.
    // neural_load_snapshot rejects any file whose embedded snapshot_sha256
    // does not match the loaded model, and the saved digest is exactly
    // neural_snapshot_sha256(snapshot), so a successful load here already
    // proves the round-trip.
    (void)neural_load_snapshot(
        file(stage.path(), model_manifest).string(),
        file(stage.path(), model_weights).string());
    stage.publish(destination);
}

NeuralParameterSnapshot neural_load_snapshot_atomic(
    std::string const& directory) {
    const auto published = existing_checkpoint(directory);
    return neural_load_snapshot(file(published, model_manifest).string(),
                                file(published, model_weights).string());
}

void neural_save_training_checkpoint_atomic(
    NeuralTrainingSession const& session,
    std::string const& directory) {
    const auto destination = publication_target(directory);
    StagingDirectory stage(destination);
    neural_save_training_checkpoint(
        session,
        file(stage.path(), model_manifest).string(),
        file(stage.path(), model_weights).string(),
        file(stage.path(), optimizer_weights).string());
    // This also checks the optimizer-state digest, model identity and schema:
    // neural_load_training_checkpoint recomputes training_sha256 over the
    // whole loaded state (steps, moments, snapshot digest, options) and
    // rejects any mismatch, so a successful load proves the round-trip.
    (void)neural_load_training_checkpoint(
        file(stage.path(), model_manifest).string(),
        file(stage.path(), model_weights).string(),
        file(stage.path(), optimizer_weights).string());
    stage.publish(destination);
}

NeuralTrainingSession neural_load_training_checkpoint_atomic(
    std::string const& directory) {
    const auto published = existing_checkpoint(directory);
    return neural_load_training_checkpoint(
        file(published, model_manifest).string(),
        file(published, model_weights).string(),
        file(published, optimizer_weights).string());
}

} // namespace Ariadne
