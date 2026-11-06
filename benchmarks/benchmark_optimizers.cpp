/***************************************************************************
 *            benchmark_optimizers.cpp
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
    unsigned seeds = 3u;
    unsigned max_steps = 500u;
    double target_loss = 0.001;
    std::uint64_t seed = 42u;
    double sgd_rate = 0.01;
    double adamw_rate = 0.01;
    double momentum = 0.9;
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

    result.append(make_valued_argument_parser(
        "", "seeds", "Number of paired seed workloads (1..16; default 3)",
        [&selection](String const& v)->VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(v);
            if (n == 0u || n > 16u) throw std::invalid_argument("invalid seeds");
            return [&selection,n] { selection.seeds = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "max-steps", "Time-to-target budget (1..100000; default 500)",
        [&selection](String const& v)->VoidFunction {
            const unsigned n = parse_unsigned<unsigned>(v);
            if (n == 0u || n > 100000u) throw std::invalid_argument("invalid max-steps");
            return [&selection,n] { selection.max_steps = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "target-loss", "Target half-squared-error loss (finite positive)",
        [&selection](String const& v)->VoidFunction {
            const double x = parse_finite(v,true);
            return [&selection,x] { selection.target_loss = x; };
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

enum class Variant { Sgd, SgdMomentum, AdamW };

char const* name(Variant v) {
    switch(v) {
        case Variant::Sgd: return "SGD";
        case Variant::SgdMomentum: return "SGD+momentum";
        case Variant::AdamW: return "AdamW";
        default: throw std::invalid_argument("invalid variant");
    }
}

NeuralOptimizerOptions optimizer(Options const& options, Variant v) {
    NeuralOptimizerOptions opt;
    opt.weight_decay = options.weight_decay;
    switch(v) {
        case Variant::Sgd:
            opt.kind = NeuralOptimizerKind::Sgd;
            opt.learning_rate = options.sgd_rate;
            opt.momentum = 0.0;
            break;
        case Variant::SgdMomentum:
            opt.kind = NeuralOptimizerKind::Sgd;
            opt.learning_rate = options.sgd_rate;
            opt.momentum = options.momentum;
            break;
        case Variant::AdamW:
            opt.kind = NeuralOptimizerKind::AdamW;
            opt.learning_rate = options.adamw_rate;
            break;
        default: throw std::invalid_argument("invalid variant");
    }
    return opt;
}

double checksum(NeuralTrainingSession const& trainer) {
    // Retain the returned snapshot for the full iteration: values()
    // provides a reference into the snapshot rather than owning its data.
    auto const snapshot = trainer.export_snapshot();
    double total = 0.0;
    for (double v : snapshot.values()) total += v;
    if(!std::isfinite(total)) throw std::runtime_error("invalid model checksum");
    return total;
}

struct Fixed {
    double ns_step;
    double final_loss;
    double checksum;
};

Fixed time_steps(Workload const& w, Options const& opt, Variant v) {
    NeuralTrainingSession trainer(w.initial,optimizer(opt,v));
    auto start = std::chrono::steady_clock::now();
    for(unsigned i=0u;i<opt.iterations;++i)
        (void)trainer.train_batch(w.inputs,w.targets);
    auto end = std::chrono::steady_clock::now();
    if(trainer.steps()!=static_cast<std::size_t>(opt.iterations))
        throw std::runtime_error("invalid number of training steps");
    const double ns=std::chrono::duration<double,std::nano>(end-start).count()
        / static_cast<double>(opt.iterations);
    const double loss=trainer.loss_batch(w.inputs,w.targets);
    if(!std::isfinite(ns) || ns<=0.0 || !std::isfinite(loss))
        throw std::runtime_error("invalid fixed timing");
    return {ns,loss,checksum(trainer)};
}

struct Target {
    bool reached;
    unsigned steps;
    double milliseconds;
    double final_loss;
    double checksum;
};

Target time_to_target(Workload const& w, Options const& opt, Variant v) {
    NeuralTrainingSession trainer(w.initial,optimizer(opt,v));
    double loss=trainer.loss_batch(w.inputs,w.targets);
    unsigned steps=0u;
    auto start=std::chrono::steady_clock::now();
    while(loss>opt.target_loss && steps<opt.max_steps) {
        (void)trainer.train_batch(w.inputs,w.targets);
        ++steps;
        // Include every post-step loss evaluation in time-to-target.
        loss=trainer.loss_batch(w.inputs,w.targets);
    }
    auto end=std::chrono::steady_clock::now();
    const double ms=std::chrono::duration<double,std::milli>(end-start).count();
    if(trainer.steps()!=static_cast<std::size_t>(steps) ||
       !std::isfinite(ms) || ms<0.0 || !std::isfinite(loss))
        throw std::runtime_error("invalid convergence timing");
    return {loss<=opt.target_loss,steps,ms,loss,checksum(trainer)};
}

double median(std::vector<double> times) {
    if(times.empty()) throw std::runtime_error("empty timing data");
    std::sort(times.begin(),times.end());
    return times[times.size()/2u];
}

struct Summary {
    double ns_step;
    double final_loss;
    bool reached;
    unsigned steps;
    double target_ms;
    double last_loss;
};

Summary summarize(std::vector<Fixed> const& fixed,
                  std::vector<Target> const& targets) {
    if(fixed.empty() || fixed.size()!=targets.size())
        throw std::runtime_error("unpaired timing samples");
    std::vector<double> fixed_times,target_times;
    for(std::size_t i=0u;i<fixed.size();++i) {
        if(fixed[i].final_loss!=fixed[0].final_loss ||
           fixed[i].checksum!=fixed[0].checksum ||
           targets[i].reached!=targets[0].reached ||
           targets[i].steps!=targets[0].steps ||
           targets[i].final_loss!=targets[0].final_loss ||
           targets[i].checksum!=targets[0].checksum)
            throw std::runtime_error("nondeterministic replay");
        fixed_times.push_back(fixed[i].ns_step);
        target_times.push_back(targets[i].milliseconds);
    }
    return {median(fixed_times),fixed[0].final_loss,targets[0].reached,
            targets[0].steps,median(target_times),targets[0].final_loss};
}

bool help_requested(List<String> const& args) {
    return std::find(args.begin(), args.end(), String("--help")) != args.end()
        || std::find(args.begin(), args.end(), String("-h")) != args.end();
}

} // namespace

int main(int argc, char const* argv[]) {
    using namespace Ariadne;
    ThreadManager::instance();
    Logger::instance().configuration().set_verbosity(1u);
    try {
        Options options;
        ThreadingCommandLineInterface cli;
        cli.add(CommandLineContribution("neural-optimizer-benchmark",parsers(options)));
        List<String> arguments;
        for(int i=0;i<argc;++i) arguments.append(String(argv[i]));
        if(!cli.acquire(arguments)) return help_requested(arguments) ? 0 : 1;
        if(ThreadManager::instance().concurrency()!=0u)
            throw std::invalid_argument(
                "NativeCpu training is sequential: use --concurrency 0");

        constexpr Variant variants[] = {Variant::Sgd,Variant::SgdMomentum,Variant::AdamW};
        constexpr std::size_t count=3u;
        struct Aggregate {
            std::vector<double> fixed_times,target_times;
            unsigned hits=0u;
        };
        Aggregate aggregates[count];
        LOGGING_PRINTLN("Ariadne ANN | SGD, SGD+momentum, AdamW")
        LOGGING_PRINTLN("Network 8 -> " << options.hidden_width << " Tanh -> 1 Identity"
            << " | batch=" << options.batch_size << " | paired seeds=" << options.seeds
            << " | samples=" << options.samples << " + warm-up"
            << " | fixed steps=" << options.iterations
            << " | target budget=" << options.max_steps)
        LOGGING_PRINTLN("Loss = 0.5 * sum squared output errors, averaged over samples")
        LOGGING_PRINTLN("Target <= " << options.target_loss << " | SGD lr=" << options.sgd_rate
            << " | momentum=" << options.momentum << " | AdamW lr=" << options.adamw_rate
            << " | decay=" << options.weight_decay)
        LOGGING_PRINTLN("Fixed-step timing excludes repeated loss checks;"
            " time-to-target timing includes post-step checks.")
        for(unsigned si=0u;si<options.seeds;++si) {
            Options seeded=options;
            seeded.seed += static_cast<std::uint64_t>(si);
            auto w=make_workload(seeded);
            NeuralTrainingSession baseline(w.initial,NeuralOptimizerOptions{});
            const double initial_loss=baseline.loss_batch(w.inputs,w.targets);
            std::vector<Fixed> fixed[count];
            std::vector<Target> target[count];
            for(unsigned sample=0u;sample<=options.samples;++sample) {
                for(std::size_t offset=0u;offset<count;++offset) {
                    const std::size_t k=(offset+static_cast<std::size_t>(sample)
                        +static_cast<std::size_t>(si))%count;
                    const auto f=time_steps(w,options,variants[k]);
                    const auto t=time_to_target(w,options,variants[k]);
                    if(sample>0u) {
                        fixed[k].push_back(f);
                        target[k].push_back(t);
                    }
                }
            }
            LOGGING_PRINTLN(std::string(95u,'='))
            LOGGING_PRINTLN("Seed=" << seeded.seed << " | initial loss="
                << std::setprecision(10) << initial_loss)
            LOGGING_PRINTLN(std::left << std::setw(16) << "Optimizer" << std::right
                << std::setw(14) << "ns/step" << std::setw(16) << "fixed loss"
                << std::setw(14) << "target step" << std::setw(17) << "target ms"
                << std::setw(18) << "last loss")
            for(std::size_t k=0u;k<count;++k) {
                const Summary x=summarize(fixed[k],target[k]);
                aggregates[k].fixed_times.push_back(x.ns_step);
                if(x.reached) {
                    ++aggregates[k].hits;
                    aggregates[k].target_times.push_back(x.target_ms);
                }
                LOGGING_PRINTLN(std::left << std::setw(16) << name(variants[k])
                    << std::right << std::fixed << std::setprecision(3)
                    << std::setw(14) << x.ns_step
                    << std::setw(16) << std::setprecision(9) << x.final_loss
                    << std::setw(14) << (x.reached ? std::to_string(x.steps)
                        : ">" + std::to_string(options.max_steps))
                    << std::setw(17) << (x.reached ? std::to_string(x.target_ms)
                        : std::string("censored"))
                    << std::setw(18) << x.last_loss)
            }
        }
        LOGGING_PRINTLN(std::string(95u,'='))
        LOGGING_PRINTLN("Medians across paired seeds (unreached target runs are censored)")
        for(std::size_t k=0u;k<count;++k) {
            auto const& a=aggregates[k];
            LOGGING_PRINTLN(name(variants[k]) << ": median fixed-step time "
                << std::fixed << std::setprecision(3) << median(a.fixed_times)
                << " ns/step | target reached " << a.hits << "/" << options.seeds)
            if(a.hits==options.seeds) {
                LOGGING_PRINTLN("  median time-to-target across seeds = "
                    << std::setprecision(3) << median(a.target_times) << " ms")
            } else {
                LOGGING_PRINTLN("  time-to-target aggregate not reported"
                    " because some paired runs were censored.")
            }
        }
        LOGGING_PRINTLN("Every trial restarts from the same seeded model/data;"
            " algorithm order rotates; a warm-up is excluded.")
        LOGGING_PRINTLN("Time-to-target includes loss evaluations after every update;"
            " fixed-step timing does not. No CI speed or convergence threshold.")
        return 0;
    } catch(std::exception const& e) {
        LOGGING_PRINTLN("benchmark_optimizers: " << e.what())
        return 1;
    }
}
