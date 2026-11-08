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
#include "ariadne-benchmarking.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace Ariadne;
namespace Bench=Ariadne::Benchmarking;

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
            const unsigned n = Bench::parse_unsigned<unsigned>(value);
            if (n == 0u || n > 100000u) throw std::invalid_argument("iterations out of range");
            return [&selection,n] { selection.iterations = n; };
        }));
    result.append(make_valued_argument_parser(
        "n", "samples", "Odd measured samples (3..51); configure warm-ups with --warmups",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = Bench::parse_unsigned<unsigned>(value);
            if (n < 3u || n > 51u || (n & 1u) == 0u)
                throw std::invalid_argument("samples must be odd, 3..51");
            return [&selection,n] { selection.samples = n; };
        }));
    result.append(make_valued_argument_parser(
        "b", "batch-size", "Examples per identical training mini-batch (1..1024)",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = Bench::parse_unsigned<unsigned>(value);
            if (n == 0u || n > 1024u) throw std::invalid_argument("batch size out of range");
            return [&selection,n] { selection.batch_size = n; };
        }));
    result.append(make_valued_argument_parser(
        "w", "width", "Number of Tanh hidden units (1..1024)",
        [&selection](String const& value) -> VoidFunction {
            const unsigned n = Bench::parse_unsigned<unsigned>(value);
            if (n == 0u || n > 1024u) throw std::invalid_argument("hidden width out of range");
            return [&selection,n] { selection.hidden_width = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "seed", "Seed for the shared initial network and data (default: 42)",
        [&selection](String const& value) -> VoidFunction {
            const auto seed = Bench::parse_unsigned<std::uint64_t>(value);
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
            const unsigned n = Bench::parse_unsigned<unsigned>(v);
            if (n == 0u || n > 16u) throw std::invalid_argument("invalid seeds");
            return [&selection,n] { selection.seeds = n; };
        }));
    result.append(make_valued_argument_parser(
        "", "max-steps", "Time-to-target budget (1..100000; default 500)",
        [&selection](String const& v)->VoidFunction {
            const unsigned n = Bench::parse_unsigned<unsigned>(v);
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

enum class OptimizerVariant { Sgd, SgdMomentum, AdamW };

char const* name(OptimizerVariant v) {
    switch(v) {
        case OptimizerVariant::Sgd: return "SGD";
        case OptimizerVariant::SgdMomentum: return "SGD+momentum";
        case OptimizerVariant::AdamW: return "AdamW";
        default: throw std::invalid_argument("invalid variant");
    }
}

NeuralOptimizerOptions optimizer(Options const& options, OptimizerVariant v) {
    NeuralOptimizerOptions opt;
    opt.weight_decay = options.weight_decay;
    switch(v) {
        case OptimizerVariant::Sgd:
            opt.kind = NeuralOptimizerKind::Sgd;
            opt.learning_rate = options.sgd_rate;
            opt.momentum = 0.0;
            break;
        case OptimizerVariant::SgdMomentum:
            opt.kind = NeuralOptimizerKind::Sgd;
            opt.learning_rate = options.sgd_rate;
            opt.momentum = options.momentum;
            break;
        case OptimizerVariant::AdamW:
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

// One complete optimizer training trial is one timed Runner operation.
// Preparation (including the seeded trainer) and post-trial correctness
// checks are both outside the timing window.
struct Fixed {
    double ns_step;
    double final_loss;
    double checksum;
};

Fixed time_steps(Workload const& w,Options const& options,OptimizerVariant v,
                 Bench::Runner const& runner) {
    std::unique_ptr<NeuralTrainingSession> trainer;
    double loss=0.0;
    double digest=0.0;
    auto const samples=runner.run(std::string(name(v))+"/fixed",1u,
        [&] {
            trainer=std::make_unique<NeuralTrainingSession>(
                w.initial,optimizer(options,v));
        },
        [&] {
            for(unsigned step=0u;step<options.iterations;++step)
                (void)trainer->train_batch(w.inputs,w.targets);
            Bench::keep_alive(*trainer);
        },
        [&] {
            if(trainer->steps()!=static_cast<std::size_t>(options.iterations))
                throw std::runtime_error("invalid number of training steps");
            loss=trainer->loss_batch(w.inputs,w.targets);
            if(!std::isfinite(loss))
                throw std::runtime_error("nonfinite fixed-step loss");
            digest=checksum(*trainer);
        },
        static_cast<double>(options.iterations));
    double const ns=samples.values().at(0u);
    if(!std::isfinite(ns)||ns<=0.0)
        throw std::runtime_error("invalid fixed-step timing");
    return {ns,loss,digest};
}

struct Target {
    bool reached;
    unsigned steps;
    double milliseconds;
    double final_loss;
    double checksum;
};

Target time_to_target(Workload const& w,Options const& options,OptimizerVariant v,
                      Bench::Runner const& runner) {
    std::unique_ptr<NeuralTrainingSession> trainer;
    double loss=0.0;
    double digest=0.0;
    unsigned steps=0u;
    auto const samples=runner.run(std::string(name(v))+"/target",1u,
        [&] {
            trainer=std::make_unique<NeuralTrainingSession>(
                w.initial,optimizer(options,v));
            loss=trainer->loss_batch(w.inputs,w.targets);
            steps=0u;
        },
        [&] {
            while(loss>options.target_loss && steps<options.max_steps) {
                (void)trainer->train_batch(w.inputs,w.targets);
                ++steps;
                // Include EVERY post-step loss evaluation in the timer.
                loss=trainer->loss_batch(w.inputs,w.targets);
            }
            Bench::keep_alive(*trainer);
        },
        [&] {
            if(trainer->steps()!=static_cast<std::size_t>(steps) ||
               !std::isfinite(loss))
                throw std::runtime_error("invalid convergence result");
            digest=checksum(*trainer);
        });
    double const ms=samples.values().at(0u)/1e6;
    if(!std::isfinite(ms)||ms<0.0)
        throw std::runtime_error("invalid convergence timing");
    return {loss<=options.target_loss,steps,ms,loss,digest};
}

// The original replay checks compare the paired samples exactly. Keep
// that correctness property independent of the selected display columns.
struct Summary {
    Bench::Statistics fixed_timing;
    Bench::Statistics target_timing;
    double final_loss;
    bool reached;
    unsigned steps;
    double last_loss;
};

Summary summarize_trials(std::vector<Fixed> const& fixed,
                         std::vector<Target> const& targets) {
    if(fixed.empty()||fixed.size()!=targets.size())
        throw std::runtime_error("unpaired timing samples");
    Bench::Samples fixed_times,target_times;
    for(std::size_t i=0u;i<fixed.size();++i) {
        if(fixed[i].final_loss!=fixed[0].final_loss ||
           fixed[i].checksum!=fixed[0].checksum ||
           targets[i].reached!=targets[0].reached ||
           targets[i].steps!=targets[0].steps ||
           targets[i].final_loss!=targets[0].final_loss ||
           targets[i].checksum!=targets[0].checksum)
            throw std::runtime_error("nondeterministic replay");
        fixed_times.add(fixed[i].ns_step);
        target_times.add(targets[i].milliseconds*1e6);
    }
    return {Bench::summarize(fixed_times),Bench::summarize(target_times),
            fixed[0].final_loss,targets[0].reached,targets[0].steps,
            targets[0].final_loss};
}

} // namespace

int main(int argc,char const* argv[]) {
    try {
        Options options;
        Bench::CommandLine cli;
        ThreadManager::instance();
        cli.add(threading_command_line_contribution());
        cli.add(CommandLineContribution("neural-optimizer-benchmark",
                                        parsers(options)));
        if(!cli.acquire(argc,argv))return 0;
        if(ThreadManager::instance().concurrency()!=0u)
            throw std::invalid_argument(
                "NativeCpu training is sequential: use --concurrency 0");

        constexpr OptimizerVariant variants[]={
            OptimizerVariant::Sgd,OptimizerVariant::SgdMomentum,OptimizerVariant::AdamW};
        constexpr std::size_t count=3u;
        struct Aggregate {
            Bench::Samples fixed_times,target_times;
            unsigned hits=0u;
        };
        Aggregate aggregates[count];

        // The optimizer order still rotates across paired seeds AND trials.
        // Runner uses zero internal warm-ups here because that rotation must
        // also encompass the CLI-selected global warm-up trials.
        Bench::Runner runner(0u,options.seeds*count);
        Bench::Report fixed_report("ANN fixed-step timings (ns/step)",
                                  cli.summary_fields());
        fixed_report.set_dimension_columns({
            {"Seed"},{"Optimizer"},{"Fixed loss",0u,Bench::TableAlignment::Right},
            {"Target steps",0u,Bench::TableAlignment::Right},
            {"Target ms",0u,Bench::TableAlignment::Right},
            {"Last loss",0u,Bench::TableAlignment::Right}
        });
        auto rules=Bench::TableOptions::ruled();
        rules.group_columns={0u};
        fixed_report.set_table_options(rules);

        Bench::Report target_report("ANN time-to-target timings (ns)",
                                   cli.summary_fields());
        target_report.set_dimension_columns({
            {"Seed"},{"Optimizer"},{"Reached"},{"Steps",0u,Bench::TableAlignment::Right}
        });
        target_report.set_table_options(rules);

        LOGGING_PRINTLN("Ariadne ANN | SGD, SGD+momentum, AdamW")
        LOGGING_PRINTLN("Network 8 -> "<<options.hidden_width
            <<" Tanh -> 1 Identity | batch="<<options.batch_size
            <<" | paired seeds="<<options.seeds
            <<" | samples="<<options.samples
            <<" | warmups="<<cli.warmups()
            <<" | fixed steps="<<options.iterations
            <<" | target budget="<<options.max_steps)
        LOGGING_PRINTLN("Loss = 0.5 * sum squared output errors, averaged over samples")
        LOGGING_PRINTLN("Target <= "<<options.target_loss
            <<" | SGD lr="<<options.sgd_rate
            <<" | momentum="<<options.momentum
            <<" | AdamW lr="<<options.adamw_rate
            <<" | decay="<<options.weight_decay)
        LOGGING_PRINTLN("Fixed-step timing excludes repeated loss checks;"
            " time-to-target includes post-step checks.")

        for(unsigned si=0u;si<options.seeds;++si) {
            Options seeded=options;
            seeded.seed+=static_cast<std::uint64_t>(si);
            Workload const workload=make_workload(seeded);
            NeuralTrainingSession baseline(
                workload.initial,NeuralOptimizerOptions{});
            double const initial_loss=baseline.loss_batch(
                workload.inputs,workload.targets);
            LOGGING_PRINTLN("Seed="<<seeded.seed
                <<" | initial loss="<<Bench::Table::number(initial_loss,10u))
            std::vector<Fixed> fixed[count];
            std::vector<Target> target[count];
            if(options.samples>std::numeric_limits<unsigned>::max()-cli.warmups())
                throw std::overflow_error("sample and warm-up count overflow");
            unsigned const total=options.samples+cli.warmups();
            for(unsigned sample=0u;sample<total;++sample) {
                for(std::size_t offset=0u;offset<count;++offset) {
                    std::size_t const k=(
                        offset+static_cast<std::size_t>(sample)
                        +static_cast<std::size_t>(si))%count;
                    Fixed const f=time_steps(workload,options,variants[k],runner);
                    Target const t=time_to_target(workload,options,variants[k],runner);
                    if(sample>=cli.warmups()) {
                        fixed[k].push_back(f);
                        target[k].push_back(t);
                    }
                }
            }
            for(std::size_t k=0u;k<count;++k) {
                Summary const row=summarize_trials(fixed[k],target[k]);
                aggregates[k].fixed_times.add(row.fixed_timing.median);
                if(row.reached) {
                    ++aggregates[k].hits;
                    aggregates[k].target_times.add(row.target_timing.median);
                }
                std::string const seed=std::to_string(seeded.seed);
                std::string const optimizer_name=name(variants[k]);
                fixed_report.add_row({
                    seed,optimizer_name,
                    Bench::Table::number(row.final_loss,9u),
                    row.reached?std::to_string(row.steps)
                               : ">"+std::to_string(options.max_steps),
                    row.reached?Bench::Table::number(
                                      row.target_timing.median/1e6,3u)
                               : "censored",
                    Bench::Table::number(row.last_loss,9u)
                },row.fixed_timing);
                target_report.add_row({
                    seed,optimizer_name,row.reached?"yes":"no",
                    std::to_string(row.steps)
                },row.target_timing);
            }
        }
        fixed_report.print();
        target_report.print();

        Bench::Table aggregated({
            {"Optimizer"},
            {"Median ns/step",0u,Bench::TableAlignment::Right},
            {"Target hits",0u,Bench::TableAlignment::Right},
            {"Median target ms",0u,Bench::TableAlignment::Right}
        },Bench::TableOptions::ruled());
        for(std::size_t k=0u;k<count;++k) {
            Aggregate const& group=aggregates[k];
            double const fixed_ns=Bench::summarize(group.fixed_times).median;
            std::string const target_ms=group.hits==options.seeds
                ? Bench::Table::number(
                    Bench::summarize(group.target_times).median/1e6,3u)
                : "censored";
            aggregated.add_row({
                name(variants[k]),Bench::Table::number(fixed_ns),
                std::to_string(group.hits)+"/"+std::to_string(options.seeds),
                target_ms
            });
        }
        LOGGING_PRINTLN("Medians across paired seeds (unreached targets are censored)")
        LOGGING_PRINTLN(aggregated.str())
        LOGGING_PRINTLN("Every trial restarts from the same seeded model/data;"
            " algorithm order rotates; CLI warm-ups are excluded.")
        LOGGING_PRINTLN("Time-to-target includes loss evaluations after every update;"
            " fixed-step timing does not. No CI speed or convergence threshold.")
        return 0;
    } catch(std::exception const& error) {
        LOGGING_PRINTLN("benchmark_optimizers: "<<error.what())
        return 1;
    }
}
