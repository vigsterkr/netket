# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "dynamics",
    "exact",
    "graph",
    "hilbert",
    "machine",
    "operator",
    "optimizer",
    "random",
    "sampler",
    "stats",
    # "supervised",
    "utils",
    "variational",
]

# First import jax stuff
from . import utils

if utils.mpi_available and utils.jax_available:
    import pyximport
    import subprocess
    import mpi4py

    print("Using XLA-MPI interop")
    # Get mpi compiler
    config = mpi4py.get_config()
    # Detect compile flags and MPI header files
    cmd_compile = [config["mpicc"], "--showme:compile"]
    out_compile = subprocess.Popen(cmd_compile, stdout=subprocess.PIPE)

    compile_flags, _ = out_compile.communicate()

    # Push all include flags in a list
    include_dirs = [p.decode()[2:] for p in compile_flags.split()]
    include_dirs.append(mpi4py.get_include())

    # mokeypatch
    pyximport.install(setup_args={"include_dirs": include_dirs})

    from .cython import mpi_xla_bridge
    from jax.lib import xla_client

    for name, fn in mpi_xla_bridge.cpu_custom_call_targets.items():
        xla_client.register_cpu_custom_call_target(name, fn)

    utils.xla_mpi_available = True


from . import (
    dynamics,
    exact,
    graph,
    hilbert,
    logging,
    machine,
    operator,
    optimizer,
    random,
    sampler,
    stats,
    # supervised,
    utils,
    variational,
    _exact_dynamics,
    _vmc,
    _steadystate,
)

# Main applications
from ._vmc import Vmc
from ._qsr import Qsr
from ._steadystate import SteadyState

from .vmc_common import (
    tree_map as _tree_map,
    trees2_map as _trees2_map,
)
