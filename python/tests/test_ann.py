# ***************************************************************************
# *            python/tests/test_ann.py
# *
# *  Copyright  2026  Luca Geretti
# *
# ***************************************************************************
#
# This file is part of Ariadne ANN.
#
# Ariadne ANN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ariadne ANN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ariadne ANN.  If not, see <https://www.gnu.org/licenses/>.

from pyariadne_ann import ann_version


def test_ann_bootstrap():
    assert ann_version() == "0.1.0"
