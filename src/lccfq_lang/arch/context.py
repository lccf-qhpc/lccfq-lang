"""
Filename: context.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides the definition for a quantum circuit as a sequence
    of gates on a number of qubits. Circuits utilize a number of qubits
    with no assumption about the mapping, which is provided by the machine
    model (`mach`).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from .instruction import Instruction
from .register import QRegister, CRegister, QContext
from .error import UnknownCompilerPass
from .protocol import Backend
from typing import List, Dict, Callable, Tuple


@dataclass
class CompilerPass:
    """A single named stage in the compilation pipeline."""
    name: str
    transform: Callable[[list], list]


class CompilationPipeline:
    """An ordered sequence of compiler passes."""

    def __init__(self, passes: List[CompilerPass]):
        self.passes = passes

    def run(self, instructions: List[Instruction], last_pass: str) -> Tuple[str, list]:
        """Run passes up to and including last_pass.

        :param instructions: raw instruction list
        :param last_pass: name of the final pass to execute
        :return: (pass_name, result) of the last pass that ran
        """
        program = instructions

        for cpass in self.passes:
            program = cpass.transform(program)

            if cpass.name == last_pass:
                return cpass.name, program

        raise UnknownCompilerPass(last_pass)


class Circuit:
    """Implementation of a quantum circuit based on instructions.

    A circuit is an atomic program composed of gates and measurements at the
    end. Circuits, after their context closes, generate code that goes into
    a circuit description to the respective backend.
    """

    def __init__(self,
                 qreg: QRegister,
                 creg: CRegister,
                 qpu: Backend = None,
                 shots: int = 1000,
                 verbose: bool = False,
                 opt_level: int = 0,
                 opt_passes: list[str] | None = None,
                 report: bool = False):
        """Create a new circuit.

        :param qreg: quantum register
        :param creg: classical register
        :param qpu: QPU backend
        :param shots: number of shots
        :param verbose: trigger verbose output
        :param opt_level: arch-level optimization level (0..3). Ignored
            when opt_passes is not None. Default 0 (no arch_opt).
        :param opt_passes: explicit list of arch pass names; overrides opt_level.
            Use [] to explicitly disable arch_opt while still using the
            explicit-mode contract.
        :param report: when True, populate :attr:`opt_report` after the
            ``with`` block exits. Default False (zero-overhead).
        """
        if not isinstance(opt_level, int) or opt_level not in (0, 1, 2, 3):
            raise ValueError(
                f"Circuit: opt_level must be one of (0, 1, 2, 3), got {opt_level!r}"
            )
        if opt_passes is not None:
            if not isinstance(opt_passes, list) or not all(
                isinstance(n, str) for n in opt_passes
            ):
                raise TypeError("Circuit: opt_passes must be None or list[str]")
        if not isinstance(report, bool):
            raise TypeError(
                f"Circuit: report must be a bool, got {type(report).__name__}"
            )

        self.qreg = qreg
        self.creg = creg
        self.qpu = qpu
        self.shots = shots
        self.verbose = verbose
        self.instructions: List[Instruction] = list()
        self._opt_records = []
        self._opt_groups_meta: dict[str, tuple] = {}
        self._opt_level = opt_level
        self._opt_passes = opt_passes
        self._report = report
        self.opt_report: dict | None = None

    def results(self) -> Dict[str, int]:
        return self.creg.data

    def frequencies(self):
        return self.creg.frequencies()

    def _handle_pass(self, program: list, cpass: str) -> None:
        """Handle compiler pass termination

        :param program: list of entities in different stages of processing composing a program
        :param cpass: current compilation pass
        :return: nothing
        """
        if self.verbose:
            print(f"Stage: {cpass}")
            print(program)
            print("\n\n")

        if cpass == "executed":
            result = self.qpu.exec_circuit(program, self.shots)
        else:
            result = {format(i, f"0{self.creg.bit_count}b"): -1 for i in range(2 ** self.creg.bit_count)}

        self.creg.absorb(result)

    def __rshift__(self, instr: Instruction):
        """Add a new instruction to the circuit using the `>>` operator. This
        removes verbosity.

        :param instr: instruction to add
        :return: none
        """

        # We try to catch errors as early as they appear, which is
        # when these are included in the code.
        challenged = self.qreg.challenge(instr, QContext.CIRCUIT)
        self.instructions.append(challenged)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, equivalent to generating the circuit and sending it to the
        backend. This allows multiple circuits to have multiple backends by construction.

        :param exc_type: exception type, if any
        :param exc_val: exception value, if any
        :param exc_tb: exception traceback, if any
        :return: False to propagate exceptions, True on success
        """
        if exc_type is not None:
            return False

        last_pass = self.qpu.last_pass

        # Phase 5: capture pipeline-entry program & cost so the report sees the
        # raw input. Cost.measure is cheap (depth + counts + estimated_error).
        if self._report:
            from lccfq_lang.opt.cost import Cost
            pipeline_input = list(self.instructions)
            pipeline_in_cost = Cost.measure(
                pipeline_input,
                kind="arch",
                qpu_config=self.qpu.config,
            )
        else:
            pipeline_input = None
            pipeline_in_cost = None

        if last_pass == "parsed":
            self._handle_pass(self.instructions, "parsed")
            if self._report:
                # Empty-pipeline report: no groups, totals show identity.
                self.opt_report = self._build_opt_report(
                    program_in=pipeline_input,
                    program_out=self.instructions,
                    in_cost=pipeline_in_cost,
                    out_cost=pipeline_in_cost,   # parsed = no transformation
                    last_pass="parsed",
                    effective_strategy=self.qreg.mapping.routing_strategy,
                    groups_meta={},
                )
            return True

        # Deferred imports: keep arch/ free of eager opt/ dependency at module load.
        from lccfq_lang.opt import PassContext, PassManager
        from lccfq_lang.opt.builtin.lower_passes import (
            build_lowering_groups, slice_groups_for,
        )
        from lccfq_lang.opt.builtin.routing import LayoutSelection

        # Phase 4: resolve effective routing strategy.
        # opt_level >= 2 forces "sabre_lite" regardless of mapping default.
        base_strategy = self.qreg.mapping.routing_strategy

        if self._opt_level >= 2:
            effective_strategy = "sabre_lite"
        else:
            effective_strategy = base_strategy

        # Optionally rebind qreg.mapping with an improved layout for this run.
        qreg_for_run = self.qreg

        if effective_strategy == "sabre_lite":
            new_layout = LayoutSelection.compute_layout(
                program=self.instructions,
                topology=self.qreg.mapping.topology,
                isa=self.qpu.isa,
                initial_layout=dict(self.qreg.mapping.mapping),
            )
            new_mapping = self.qreg.mapping.with_layout(new_layout)
            qreg_for_run = self.qreg.rebind_mapping(new_mapping)

        groups = slice_groups_for(
            last_pass,
            build_lowering_groups(
                qreg_for_run,
                self.qpu,
                opt_level=self._opt_level,
                opt_passes=self._opt_passes,
                routing_strategy=effective_strategy,
            ),
        )
        ctx = PassContext(
            qpu_config=self.qpu.config,
            isa=self.qpu.isa,
            mapping=qreg_for_run.mapping,
            topology=qreg_for_run.mapping.topology,
        )
        program, records, groups_meta = PassManager(groups).run(self.instructions, ctx)
        self._opt_records = records
        self._opt_groups_meta = groups_meta
        self._handle_pass(program, last_pass)

        if self._report:
            from lccfq_lang.opt.cost import Cost
            kind_after = (
                "arch"
                if last_pass in ("parsed", "mapped", "swapped",
                                 "expanded", "arch_optimized")
                else "mach"
            )
            pipeline_out_cost = Cost.measure(
                program,
                kind=kind_after,
                qpu_config=self.qpu.config,
            )
            self.opt_report = self._build_opt_report(
                program_in=pipeline_input,
                program_out=program,
                in_cost=pipeline_in_cost,
                out_cost=pipeline_out_cost,
                last_pass=last_pass,
                effective_strategy=effective_strategy,
                groups_meta=self._opt_groups_meta,
            )

        return True

    def _build_opt_report(
        self,
        *,
        program_in,
        program_out,
        in_cost,
        out_cost,
        last_pass: str,
        effective_strategy: str,
        groups_meta: dict | None = None,
    ) -> dict:
        """Assemble the structured opt_report dict from self._opt_records and
        pre/post pipeline costs.

        Parameters
        ----------
        groups_meta:
            Mapping from group name to ``(group_cost_before, group_cost_after)``
            for fixpoint groups, as returned by :meth:`PassManager.run` (Perf #1).
            When provided, fixpoint group boundaries use these full-Cost values
            instead of ``rs[0].cost_before`` / ``rs[-1].cost_after``, which may
            have ``depth=None`` under Perf #1.  Linear groups are absent from
            this dict and continue to derive boundaries from records.

        Side-effect-free; safe to call from __exit__ exactly once.
        """
        if groups_meta is None:
            groups_meta = {}

        # Group records by group_name preserving first-seen order.
        groups_in_order: list[str] = []
        by_group: dict[str, list] = {}
        for r in self._opt_records:
            if r.group_name not in by_group:
                groups_in_order.append(r.group_name)
                by_group[r.group_name] = []
            by_group[r.group_name].append(r)

        # Build per-group entries.
        group_entries = []
        for gname in groups_in_order:
            rs = by_group[gname]
            # Mode inference: if any record carries iteration > 0, the group ran
            # in fixpoint mode. Otherwise linear. (PassManager guarantees iteration=0
            # for linear groups and >=0 for fixpoint groups.)
            max_it = max(r.iteration for r in rs)
            mode = "fixpoint" if max_it > 0 else _infer_group_mode(gname)
            # iterations = max_it + 1 for fixpoint; 1 for linear
            iterations = max_it + 1 if mode == "fixpoint" else 1

            # For fixpoint groups, use the full-Cost group-boundary values from
            # groups_meta (real depth); for linear groups, fall back to the
            # first/last records (which use full Cost.measure under Perf #1 C.2).
            if gname in groups_meta:
                group_in_cost, group_out_cost = groups_meta[gname]
            else:
                group_in_cost = rs[0].cost_before
                group_out_cost = rs[-1].cost_after
            group_entries.append({
                "name": gname,
                "mode": mode,
                "iterations": iterations,
                "passes": [
                    {
                        "name": r.pass_name,
                        "iteration": r.iteration,
                        "cost_before": _cost_to_dict(r.cost_before),
                        "cost_after": _cost_to_dict(r.cost_after),
                        "delta_seconds": r.delta_seconds,
                    }
                    for r in rs
                ],
                "cost_before": _cost_to_dict(group_in_cost),
                "cost_after": _cost_to_dict(group_out_cost),
                "scalarized_delta": (
                    group_in_cost.scalarize() - group_out_cost.scalarize()
                ),
            })

        total_seconds = sum(r.delta_seconds for r in self._opt_records)

        return {
            "opt_level": self._opt_level,
            "opt_passes": (
                list(self._opt_passes) if self._opt_passes is not None else None
            ),
            "routing_strategy": effective_strategy,
            "last_pass": last_pass,
            "groups": group_entries,
            "totals": {
                "cost_before": _cost_to_dict(in_cost),
                "cost_after": _cost_to_dict(out_cost),
                "scalarized_delta": in_cost.scalarize() - out_cost.scalarize(),
                "total_seconds": total_seconds,
            },
        }


class Test:
    """Implementaiton of a test context for LCCFQ.

    A test context is either a specialized instruction corresponding to hardware-level
    primitives (e.g., power Rabi, resonator spectroscopy) or the use of a single quantum
    instruction as a test.

    """
    def __init__(self,
                 qreg: QRegister,
                 accum: Dict[int,Dict[str,float]],
                 qpu: Backend = None,
                 verbose=False):
        """Create a new test.

        :param qreg: quantum register
        :param accum: accumulator for test results
        :param qpu: QPU backend
        :param verbose: trigger verbose output
        """
        self.qreg = qreg
        self.accum = accum
        self.qpu = qpu
        self.verbose = verbose
        self.instructions: List[Instruction] = list()

    def __rshift__(self, instr: Instruction):
        """Add a new instruction to the circuit using the `>>` operator. This
        removes verbosity.

        :param instr: instruction to add
        :return: none
        """

        # We try to catch errors as early as they appear, which is
        # when these are included in the code.
        challenged = self.qreg.challenge(instr, QContext.TEST)
        self.instructions.append(challenged)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exiting the context leaves in the dictionary at the start
        the corresponding output per instruction type.

        :param exc_type: none
        :param exc_val: none
        :param exc_tb: none
        :return: nothing
        """
        for i, instruction in enumerate(self.instructions):
            res = self.qpu.exec_single(instruction, instruction.shots)
            self.accum[i] = res

class Control:
    # TODO: control here means a change to the QPU state or its defining thresholds.
    pass


# ---------------------------------------------------------------------------
# Module-level helpers for opt_report construction (Phase 5)
# ---------------------------------------------------------------------------

def _cost_to_dict(c) -> dict:
    """Stable JSON-friendly snapshot of a Cost dataclass.

    Includes the scalarized score so report consumers don't have to recompute it.
    """
    return {
        "depth": c.depth,
        "count_1q": c.count_1q,
        "count_2q": c.count_2q,
        "count_native_2q": c.count_native_2q,
        "estimated_error": c.estimated_error,
        "scalarized": c.scalarize(),
    }


# Static map from group name to default mode. Used only when no record's
# iteration is > 0 (so we can't disambiguate from telemetry alone).
_GROUP_DEFAULT_MODE = {
    "lower_map":       "linear",
    "lower_swap":      "linear",
    "lower_expand":    "linear",
    "arch_opt":        "fixpoint",
    "lower_transpile": "linear",
    "mach_opt":        "fixpoint",
}


def _infer_group_mode(group_name: str) -> str:
    """Return the canonical PassGroup mode for a known lowering group name.

    For unknown groups (user-extended pipelines not envisioned in Phase 5),
    default to 'linear' — the safer choice for downstream consumers that
    treat 'fixpoint' as more expensive.
    """
    return _GROUP_DEFAULT_MODE.get(group_name, "linear")
