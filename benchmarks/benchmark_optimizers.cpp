/***************************************************************************
 *            benchmark_optimizers.cpp
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
#include "threading/thread_manager.hpp"
#include "threading/threading_command_line_interface.hpp"
#include "logging/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace Ariadne;

struct Options {
    unsigned iterations = 50u;
    unsigned samples = 7u;
    unsigned batch_size = 16u;
    unsigned hidden_width = 32u;
    std::uint64_t seed = 42u;
    double sgd_rate = 0.01;
    double adamw_rate = 0.01;
    double momentum = 0.0;
    double weight_decay = 0.0;
};

template<class UInt>
UInt parse_unsigned(String const& value) {
    const std::string text(value);
    if (text.empty() || text.front() == '-')
        throw std::invalid_argument("expected an unsigned integer");
    std::size_t used = 0u;
    const unsigned long long number = std::stoull(text, &used);
    if (used != text.size() || number > std::numeric_limits<UInt>::max())
        throw std::invalid_argument("unsigned integer out of range");
    return static_cast<UInt>(number);
}

double parse_finite(String const& value, bool strictly_positive) {
    const std::string text(value);
    std::size_t used = 0u;
    const double number = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(number) ||
        (strictly_positive ? number <= 0.0 : number < 0.0))
        throw std::invalid_argument("invalid finite nonnegative/positive value");
    return number;
}

List<ArgumentParser> parsers(Options& selection) {
    List<ArgumentParser> result;
    result.append(make_valued_argument_parser(
        "i", "iterations", "Training steps per timing sample (1..100000)",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(value);
            if (n == 0u || n > 100000u) throw std::invalid_argument("iterations out of range");
            return [&selection,n] { selection.iterations = n; };
        }));
    result.append(make_valued_argument_parser(
        "n", "samples", "Odd number of measured samples (3..51), plus warm-up",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(value);
            if (n < 3u || n > 51u || (n & 1u) == 0u)
                throw std::invalid_argument("samples must be odd, 3..51");
            return [&selection,n] { selection.samples = n; };
        }));
    result.append(make_valued_argument_parser(
        "b", "batch-size", "Examples per identical training mini-batch (1..1024)",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(value);
            if (n == 0u || n > 1024u) throw std::invalid_argument("batch size out of range");
            return [&selection,n] { selection.batch_size = n; };
        }));
    result.append(make_valued_argument_parser(
        "w", "width", "Number of Tanh hidden units (1..1024)",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(value);
            if (n == 0u || n > 1024u) throw std::invalid_argument("hidden width out of range");
            return [&selection,n] { selection.hidden_width = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "seed", "Seed for the shared initial network and data (default: 42)",
        [&selection](String const& value) -> VoidFunction {
            const auto seed = parse_unsigned<std::uint64_t>(value);
            return [&selection,seed] { selection.seed = seed; };
        }));
    result.append(make_valued_argument_parser(
        "", "sgd-lr", "SGD learning rate (positive finite; default: 0.01)",
        [&selection](String const& value) -> VoidFunction {
            const double n = parse_finite(value, true);
            return [&selection,n] { selection.sgd_rate = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "adamw-lr", "AdamW learning rate (positive finite; default: 0.01)",
        [&selection](String const& value) -> VoidFunction {
            const double n = parse_finite(value, true);
            return [&selection,n] { selection.adamw_rate = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "sgd-momentum", "SGD classical momentum (0 <= m < 1; default: 0)",
        [&selection](String const& value) -> VoidFunction {
            const double n = parse_finite(value, false);
            if (n >= 1.0) throw std::invalid_argument("momentum must be less than 1");
            return [&selection,n] { selection.momentum = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "weight-decay", "Weight decay (nonnegative; default: 0)",
        [&selection](String const& value) -> VoidFunction {
            const double n = parse_finite(value, false);
            return [&selection,n] { selection.weight_decay = n; };
        }));
    return result;
}

struct Workload {
    NeuralParameterSnapshot initial;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
};

Workload make_workload(Options const& options) {
    constexpr std::size_t input_dimension = 8u;
    NeuralNetwork model(input_dimension, {
        {static_cast<std::size_t>(options.hidden_width), NeuralActivation::Tanh},
        {1u, NeuralActivation::Identity}
    });
    Workload workload{neural_initialize_parameters(std::move(model), options.seed), {}, {}};
    workload.inputs.reserve(options.batch_size);
    workload.targets.reserve(options.batch_size);
    const double phase = static_cast<double>(options.seed % 101u) * 0.01;
    for (unsigned n = 0u; n < options.batch_size; ++n) {
        std::vector<double> input;
        input.reserve(input_dimension);
        for (std::size_t i = 0u; i < input_dimension; ++i) {
            const double position =
                static_cast<double>(n + 1u) * static_cast<double>(i + 2u);
            input.push_back(0.8 * std::sin(0.13 * position + phase));
        }
        const double target = 0.4 * input[0] - 0.2 * input[1] + 0.1 * input[2];
        workload.inputs.push_back(std::move(input));
        workload.targets.push_back({target});
    }
    return workload;
}

struct Measurement {
    double nanoseconds_per_step;
    double final_loss;
    double parameter_checksum;
};

Measurement run_once(Workload const& workload, Options const& selection,
                     NeuralOptimizerKind kind) {
    NeuralOptimizerOptions optimizer;
    optimizer.kind = kind;
    optimizer.learning_rate =
        kind == NeuralOptimizerKind::Sgd ? selection.sgd_rate : selection.adamw_rate;
    optimizer.weight_decay = selection.weight_decay;
    if (kind == NeuralOptimizerKind::Sgd) optimizer.momentum = selection.momentum;
    NeuralTrainingSession training(workload.initial, optimizer);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned step = 0u; step < selection.iterations; ++step)
        (void)training.train_batch(workload.inputs, workload.targets);
    const auto stop = std::chrono::steady_clock::now();

    if (training.steps() != static_cast<std::size_t>(selection.iterations))
        throw std::runtime_error("incorrect number of optimizer updates");
    const double elapsed_ns =
        std::chrono::duration<double, std::nano>(stop - start).count();
    const double ns_step = elapsed_ns / static_cast<double>(selection.iterations);
    const double final_loss = training.loss_batch(workload.inputs, workload.targets);
    double checksum = 0.0;
    for (double p : training.export_snapshot().values()) checksum += p;
    if (!std::isfinite(ns_step) || ns_step <= 0.0 ||
        !std::isfinite(final_loss) || !std::isfinite(checksum))
        throw std::runtime_error("invalid benchmark measurement");
    return {ns_step, final_loss, checksum};
}

struct Summary {
    double median_ns_step;
    double final_loss;
};

Summary summarize(std::vector<Measurement> const& measurements) {
    if (measurements.empty()) throw std::runtime_error("no benchmark measurements");
    std::vector<double> times;
    times.reserve(measurements.size());
    for (auto const& m : measurements) {
        if (m.final_loss != measurements.front().final_loss ||
            m.parameter_checksum != measurements.front().parameter_checksum)
            throw std::runtime_error("optimizer replay is not deterministic");
        times.push_back(m.nanoseconds_per_step);
    }
    std::sort(times.begin(), times.end());
    return {times[times.size()/2u], measurements.front().final_loss};
}

bool help_requested(List<String> const& args) {
    return std::find(args.begin(), args.end(), String("--help")) != args.end()
        || std::find(args.begin(), args.end(), String("-h")) != args.end();
}

} // namespace

int main(int argc, char const* argv[]) {
    using namespace Ariadne;
    // ThreadManager installs the logging thread registry; default concurrency
    // is zero. The M4 optimizer implementation itself is sequential.
    ThreadManager::instance();
    Logger::instance().configuration().set_verbosity(1u);
    try {
        Options selection;
        ThreadingCommandLineInterface cli;
        cli.add(CommandLineContribution("neural-optimizer-benchmark", parsers(selection)));
        List<String> arguments;
        for (int i = 0; i < argc; ++i) arguments.append(String(argv[i]));
        if (!cli.acquire(arguments)) return help_requested(arguments) ? 0 : 1;
        if (ThreadManager::instance().concurrency() != 0u)
            throw std::invalid_argument(
                "NativeCpu M4 training is sequential: use --concurrency 0");

        auto workload = make_workload(selection);
        NeuralTrainingSession baseline(workload.initial, NeuralOptimizerOptions{});
        const double initial_loss = baseline.loss_batch(workload.inputs, workload.targets);
        std::vector<Measurement> sgd_measurements, adamw_measurements;
        sgd_measurements.reserve(selection.samples);
        adamw_measurements.reserve(selection.samples);

        LOGGING_PRINTLN("Ariadne Neural | NativeCpu SGD vs AdamW end-to-end training")
        LOGGING_PRINTLN("Network: 8 -> " << selection.hidden_width << " (Tanh) -> 1 (Identity)"
            << " | batch=" << selection.batch_size << " | steps=" << selection.iterations
            << " | samples=" << selection.samples << " + warm-up"
            << " | seed=" << selection.seed)
        LOGGING_PRINTLN("Sequential execution: concurrency=0, MSE (mean over samples)")
        LOGGING_PRINTLN("SGD lr=" << selection.sgd_rate << " momentum=" << selection.momentum
            << " | AdamW lr=" << selection.adamw_rate
            << " | weight decay=" << selection.weight_decay)
        LOGGING_PRINTLN("Initial shared loss: " << std::setprecision(10) << initial_loss)

        // Each sample starts from precisely the same immutable model and data.
        // Sample zero is a full untimed warm-up; alternate the order of both
        // measured implementations to reduce sequential order bias.
        for (unsigned sample = 0u; sample <= selection.samples; ++sample) {
            const bool sgd_first = (sample % 2u) == 0u;
            const auto first = run_once(workload, selection,
                sgd_first ? NeuralOptimizerKind::Sgd : NeuralOptimizerKind::AdamW);
            const auto second = run_once(workload, selection,
                sgd_first ? NeuralOptimizerKind::AdamW : NeuralOptimizerKind::Sgd);
            if (sample != 0u) {
                sgd_measurements.push_back(sgd_first ? first : second);
                adamw_measurements.push_back(sgd_first ? second : first);
            }
        }

        const auto sgd = summarize(sgd_measurements);
        const auto adamw = summarize(adamw_measurements);
        LOGGING_PRINTLN(std::string(74u, '='))
        LOGGING_PRINTLN(std::left << std::setw(12) << "Optimizer"
            << std::right << std::setw(20) << "Median ns/step"
            << std::setw(21) << "Final MSE" << std::setw(19) << "Steps")
        LOGGING_PRINTLN(std::string(74u, '-'))
        LOGGING_PRINTLN(std::left << std::setw(12) << "SGD"
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(20) << sgd.median_ns_step
            << std::setw(21) << std::setprecision(10) << sgd.final_loss
            << std::setw(19) << selection.iterations)
        LOGGING_PRINTLN(std::left << std::setw(12) << "AdamW"
            << std::right << std::fixed << std::setprecision(3)
            << std::setw(20) << adamw.median_ns_step
            << std::setw(21) << std::setprecision(10) << adamw.final_loss
            << std::setw(19) << selection.iterations)
        LOGGING_PRINTLN(std::string(74u, '='))
        LOGGING_PRINTLN("AdamW / SGD time per step: " << std::setprecision(3)
            << adamw.median_ns_step / sgd.median_ns_step << "x (>1 is slower)")
        LOGGING_PRINTLN("Time includes forward, loss, backpropagation, optimizer and"
            " state/snapshot updates; excludes model/data setup and initial/final loss checks.")
        LOGGING_PRINTLN("This is fixed-step progress, NOT a matched-quality or"
            " time-to-accuracy comparison. Different learning rates are configurable.")
        LOGGING_PRINTLN("No performance threshold is enforced in CI.")
        return 0;
    } catch (std::exception const& error) {
        LOGGING_PRINTLN("benchmark_optimizers: " << error.what())
        return 1;
    }
}
