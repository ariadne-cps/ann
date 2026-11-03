/***************************************************************************
 *            tutorial/tutorial_neural.cpp
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

#include <iostream>
#include <vector>

int main() {
    using namespace Ariadne;
    NeuralNetwork network(2, {{2, NeuralActivation::Relu},
                              {1, NeuralActivation::Identity}});
    NeuralParameterSnapshot parameters(
        network, {1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0});
    NeuralExecutionSession session(parameters);

    std::cout << "Ariadne Neural " << neural_version() << '\n'
              << "Network dimensions: " << network.input_dimension()
              << " -> " << network.output_dimension() << '\n'
              << "Parameters: " << network.parameter_count() << '\n'
              << "Native CPU snapshot import: "
              << session.supports(NeuralOperation::ImportSnapshot) << '\n'
              << "Inference available: "
              << session.supports(NeuralOperation::Inference) << '\n'
              << "Single inference: " << session.infer({3.0, 1.0}).at(0) << '\n'
              << "Batch inference: " << session.infer_batch({{3.0, 1.0}, {1.0, 4.0}}).at(1).at(0) << '\n';
    return 0;
}
