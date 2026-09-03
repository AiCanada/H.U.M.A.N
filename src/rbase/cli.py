# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RBase Command Line Interface"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse

from rbase.data.msa.mmseq2_colab import add_args as _add_msa_args
from rbase.data.msa.mmseq2_colab import cli as _msa_cli
from rbase.data.pretrain_repr.openfold.make_openfold_repr import (
    add_args as _add_openfold_args,
)
from rbase.data.pretrain_repr.openfold.make_openfold_repr import (
    cli as _openfold_cli,
)
from rbase.inference import add_args as _add_generate_args
from rbase.inference import cli as _generate_cli
from rbase.train import add_args as _add_train_args
from rbase.train import cli as _train_cli
from rbase.utils import configure_stdio, get_pylogger, install_debug_hooks

configure_stdio()
install_debug_hooks()
log = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Components
# =============================================================================

def build_parser():
    main_parser = argparse.ArgumentParser(
        prog="rbase",
        description="See below for sub-commands",
        usage="%(prog)s <command> [args]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = main_parser.add_subparsers(dest="command", help="")
    subparsers.required = True

    # Query MSA
    msa_parser = subparsers.add_parser(
        "query_msa",
        help="Query MSA using ColabFold's MMSeq2 server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    msa_parser = _add_msa_args(msa_parser)
    msa_parser.set_defaults(func=_msa_cli)

    # Generate Openfold repr
    openfold_repr_parser = subparsers.add_parser(
        "openfold_repr",
        help="Make OpenFold representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    openfold_repr_parser = _add_openfold_args(openfold_repr_parser)
    openfold_repr_parser.set_defaults(func=_openfold_cli)

    # Run RBase generation
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate RBase samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate_parser = _add_generate_args(generate_parser)
    generate_parser.set_defaults(func=_generate_cli)

    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune ConfRover-base-20M on Dual Personality Fragments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser = _add_train_args(train_parser)
    train_parser.set_defaults(func=_train_cli)

    return main_parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except BaseException:
        log.exception("Uncaught exception in `rbase %s`", getattr(args, "command", "?"))
        raise

if __name__ == "__main__":
    main()
