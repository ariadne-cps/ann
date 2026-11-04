/***************************************************************************
 *            src/checkpoint.cpp
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

#include "neural/serialization.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
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
    if (fs::exists(target) || fs::is_symlink(target))
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
        static std::atomic<std::uint64_t> sequence{0u};
        const auto ticks = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        const auto index = sequence.fetch_add(1u, std::memory_order_relaxed);
        const fs::path parent = destination.has_parent_path()
            ? destination.parent_path() : fs::path(".");
        const std::string prefix = destination.filename().string()
            + ".staging." + std::to_string(ticks)
            + "." + std::to_string(index) + ".";
        for (unsigned attempt = 0u; attempt != 64u; ++attempt) {
            const auto candidate = parent / (prefix + std::to_string(attempt));
            std::error_code ec;
            if (fs::create_directory(candidate, ec)) {
                path_ = candidate;
                return;
            }
            if (ec && ec != std::errc::file_exists)
                throw fs::filesystem_error(
                    "Cannot create checkpoint staging directory", candidate, ec);
        }
        throw std::runtime_error("Cannot allocate checkpoint staging directory");
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
        if (fs::exists(destination) || fs::is_symlink(destination))
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
    const auto restored = neural_load_snapshot(
        file(stage.path(), model_manifest).string(),
        file(stage.path(), model_weights).string());
    if (neural_snapshot_sha256(snapshot) != neural_snapshot_sha256(restored))
        throw std::runtime_error("Staged model snapshot did not round-trip");
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
    // This also checks the optimizer-state digest, model identity and schema.
    const auto restored = neural_load_training_checkpoint(
        file(stage.path(), model_manifest).string(),
        file(stage.path(), model_weights).string(),
        file(stage.path(), optimizer_weights).string());
    if (restored.steps() != session.steps() ||
        neural_snapshot_sha256(restored.export_snapshot()) !=
        neural_snapshot_sha256(session.export_snapshot()))
        throw std::runtime_error("Staged training checkpoint did not round-trip");
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
