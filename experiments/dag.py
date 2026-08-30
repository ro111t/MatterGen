"""Global experiment dependency graph builder and validator.

Builds a validated directed acyclic graph (DAG) of all experiment tasks,
ensuring immutable source memory construction, strict parent artifact verification,
exact run counts, and rejection of cycles or duplicate IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from experiments.spec import (
    FIVE_CONDITIONS,
    MEMORY_ARMS,
    ExperimentSpec,
    RunSpec,
    TaskDefinition,
    TransferDeclaration,
    compute_sha256,
)


class NodeType(str, Enum):
    PREFLIGHT = "preflight"
    REFERENCE_SET_VERIFICATION = "reference_set_verification"
    SOURCE_MEMORY_RUN = "source_memory_run"
    SOURCE_MEMORY_SNAPSHOT = "source_memory_snapshot"
    TARGET_CAMPAIGN_RUN = "target_campaign_run"
    AGGREGATION = "aggregation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    QE_AUDIT_SELECTION = "qe_audit_selection"
    QE_AUDIT_EXECUTION = "qe_audit_execution"
    REPORT_GENERATION = "report_generation"


class DAGValidationError(ValueError):
    """Raised when DAG building or validation fails."""


@dataclass
class DAGNode:
    """A node in the experiment dependency graph."""
    node_id: str
    node_type: NodeType
    description: str
    dependencies: Set[str] = field(default_factory=set)
    payload: Dict[str, Any] = field(default_factory=dict)
    expected_output_path: Optional[str] = None
    expected_output_hash: Optional[str] = None
    executed: bool = False
    success: bool = False
    result_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "description": self.description,
            "dependencies": sorted(self.dependencies),
            "payload": self.payload,
            "expected_output_path": self.expected_output_path,
            "expected_output_hash": self.expected_output_hash,
            "executed": self.executed,
            "success": self.success,
            "result_hash": self.result_hash,
        }


class ExperimentDAG:
    """Directed Acyclic Graph orchestrator for benchmark experiments."""

    def __init__(self, spec: ExperimentSpec):
        self.spec = spec
        self.nodes: Dict[str, DAGNode] = {}
        self._build_graph()
        self.validate()

    def add_node(self, node: DAGNode) -> None:
        if node.node_id in self.nodes:
            raise DAGValidationError(f"Duplicate node_id '{node.node_id}' in DAG")
        self.nodes[node.node_id] = node

    def assert_dependencies_succeeded(self, node_id: str) -> None:
        """Fail closed when a parent is missing, skipped, or unsuccessful."""
        if node_id not in self.nodes:
            raise DAGValidationError(f"Unknown node '{node_id}'")
        node = self.nodes[node_id]
        for dep_id in sorted(node.dependencies):
            parent = self.nodes.get(dep_id)
            if parent is None:
                raise DAGValidationError(f"Node '{node_id}' depends on missing node '{dep_id}'")
            if not parent.executed or not parent.success:
                raise DAGValidationError(
                    f"Node '{node_id}' cannot execute: dependency '{dep_id}' did not succeed"
                )

    @staticmethod
    def hash_node_output(node: DAGNode) -> Optional[str]:
        """Hash an output artifact after successful execution."""
        if not node.expected_output_path:
            return None
        path = Path(node.expected_output_path)
        if not path.exists() or not path.is_file():
            raise DAGValidationError(
                f"Node '{node.node_id}' did not produce expected output '{path}'"
            )
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if node.expected_output_hash and digest != node.expected_output_hash:
            raise DAGValidationError(
                f"Node '{node.node_id}' output hash mismatch: expected {node.expected_output_hash}, got {digest}"
            )
        return digest

    def verify_node_output(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        digest = self.hash_node_output(node)
        if node.result_hash and digest != node.result_hash:
            raise DAGValidationError(
                f"Node '{node.node_id}' recorded result hash mismatch: "
                f"expected {node.result_hash}, got {digest}"
            )
        return True

    def mark_node(self, node_id: str, *, success: bool, result_hash: Optional[str] = None) -> None:
        """Update node state deterministically for resumable DAG execution."""
        node = self.nodes[node_id]
        node.executed = True
        node.success = bool(success)
        node.result_hash = result_hash

    def _build_graph(self) -> None:
        spec = self.spec
        output_root = Path(spec.output_root)

        # 1. Preflight Node
        preflight_id = "node_preflight"
        self.add_node(DAGNode(
            node_id=preflight_id,
            node_type=NodeType.PREFLIGHT,
            description="Verify software environment, git base commit, and non-dirty tree",
            payload={"code_commit": spec.code_commit, "schema_version": spec.schema_version},
            expected_output_path=str(output_root / "preflight.json").replace("\\", "/"),
        ))

        # 2. Reference Set Verification Nodes
        all_tasks = [spec.source_task] + list(spec.target_tasks)
        ref_node_ids = set()
        for task in all_tasks:
            ref_id = f"node_refset_{task.task_id}"
            ref_node_ids.add(ref_id)
            self.add_node(DAGNode(
                node_id=ref_id,
                node_type=NodeType.REFERENCE_SET_VERIFICATION,
                description=f"Verify immutable certified reference set for {task.task_id}",
                dependencies={preflight_id},
                payload={
                    "task_id": task.task_id,
                    "elements": task.elements,
                    "reference_set_path": task.reference_set_path,
                    "reference_set_sha256": task.reference_set_sha256,
                    "reference_set_certified": task.reference_set_certified,
                },
                expected_output_path=str(output_root / "references" / f"{task.task_id}.verified.json").replace("\\", "/"),
            ))

        # 3. Neutral Source Memory Runs & Snapshots per Master Seed
        source_snapshot_node_ids: Dict[int, str] = {}
        for seed in spec.master_seeds:
            # Source campaign run (neutral fixed policy for Li-P-S)
            source_run_id = f"run_source_{spec.source_task.task_id}_seed{seed}"
            source_run_node_id = f"node_{source_run_id}"
            source_run_dir = str(output_root / "runs" / "source" / spec.source_task.task_id / str(seed)).replace("\\", "/")
            
            self.add_node(DAGNode(
                node_id=source_run_node_id,
                node_type=NodeType.SOURCE_MEMORY_RUN,
                description=f"Run neutral source campaign for {spec.source_task.task_id} with seed {seed}",
                dependencies=ref_node_ids | {preflight_id},
                payload={
                    "run_id": source_run_id,
                    "task_id": spec.source_task.task_id,
                    "elements": spec.source_task.elements,
                    "condition": "source_neutral",
                    "memory_mode": "structured_provenance",
                    "seed": seed,
                    "proposal_budget": spec.proposals_per_run,
                    "oracle_budget": spec.oracle_budget_per_run,
                    "output_dir": source_run_dir,
                    "reference_set_path": spec.source_task.reference_set_path,
                    "reference_set_sha256": spec.source_task.reference_set_sha256,
                    "reference_set_certified": spec.source_task.reference_set_certified,
                    "source_memory_snapshot_sha256": None,
                },
                expected_output_path=f"{source_run_dir}/manifest.json",
            ))

            # Source memory snapshot creation & freezing
            snapshot_node_id = f"node_snapshot_source_seed{seed}"
            snapshot_path = str(output_root / "memory_snapshots" / f"source_{spec.source_task.task_id}_seed{seed}.db").replace("\\", "/")
            self.add_node(DAGNode(
                node_id=snapshot_node_id,
                node_type=NodeType.SOURCE_MEMORY_SNAPSHOT,
                description=f"Freeze and hash immutable CareerMemory snapshot for seed {seed}",
                dependencies={source_run_node_id},
                payload={
                    "source_run_id": source_run_id,
                    "seed": seed,
                    "source_task": spec.source_task.task_id,
                    "snapshot_path": snapshot_path,
                    # The sidecar's authorization set is part of the source
                    # node spec.  A restart must validate it before reusing a
                    # content-addressed snapshot.
                    "allowed_transfer_declarations": [
                        td.to_dict() if hasattr(td, "to_dict") else asdict(td)
                        for td in spec.transfer_declarations
                    ],
                },
                expected_output_path=snapshot_path,
            ))
            source_snapshot_node_ids[seed] = snapshot_node_id

        # 4. Target Campaign Runs (Every condition x seed x target task)
        target_run_node_ids: Set[str] = set()
        for target_task in spec.target_tasks:
            for cond in spec.conditions:
                for seed in spec.master_seeds:
                    run_id = f"run_{cond}_{target_task.task_id}_seed{seed}"
                    target_node_id = f"node_{run_id}"
                    target_run_dir = str(output_root / "runs" / cond / target_task.task_id / str(seed)).replace("\\", "/")
                    
                    dependencies = {preflight_id, f"node_refset_{target_task.task_id}"}
                    # Memory conditions depend on the immutable source snapshot for that paired seed
                    if cond in MEMORY_ARMS:
                        dependencies.add(source_snapshot_node_ids[seed])
                    
                    # Resolve transfer declaration for target task
                    t_decl = next(
                        (td for td in spec.transfer_declarations if td.target_task == target_task.task_id),
                        None
                    )
                    transfer_dict = {
                        "allowed_relationship": t_decl.allowed_relationship if t_decl else "homologous_series",
                        "source_chemical_system": t_decl.source_chemical_system if t_decl else spec.source_task.elements,
                        "target_chemical_system": t_decl.target_chemical_system if t_decl else target_task.elements,
                        "confidence": t_decl.confidence if t_decl else 0.8,
                    } if t_decl else None

                    # Memory mode resolution
                    mem_mode = "none"
                    if cond == "text_summary_memory":
                        mem_mode = "text_summary"
                    elif cond == "structured_provenance_memory":
                        mem_mode = "structured_provenance"
                    elif cond == "shuffled_memory_control":
                        mem_mode = "shuffled_control"

                    snapshot_path = str(output_root / "memory_snapshots" / f"source_{spec.source_task.task_id}_seed{seed}.db").replace("\\", "/") if cond in MEMORY_ARMS else None

                    self.add_node(DAGNode(
                        node_id=target_node_id,
                        node_type=NodeType.TARGET_CAMPAIGN_RUN,
                        description=f"Target run: {cond} on {target_task.task_id} with seed {seed}",
                        dependencies=dependencies,
                        payload={
                            "run_id": run_id,
                            "condition": cond,
                            "task_id": target_task.task_id,
                            "elements": target_task.elements,
                            "seed": seed,
                            "memory_mode": mem_mode,
                            "memory_seed": seed,
                            "memory_transfer_declaration": transfer_dict,
                            "source_memory_snapshot_path": snapshot_path,
                            "source_memory_snapshot_sha256": None,
                            "reference_set_path": target_task.reference_set_path,
                            "reference_set_sha256": target_task.reference_set_sha256,
                            "reference_set_certified": target_task.reference_set_certified,
                            "proposal_budget": spec.proposals_per_run,
                            "oracle_budget": spec.oracle_budget_per_run,
                            "output_dir": target_run_dir,
                        },
                        expected_output_path=f"{target_run_dir}/manifest.json",
                    ))
                    target_run_node_ids.add(target_node_id)

        # 5. Aggregation Node
        agg_node_id = "node_aggregation"
        agg_candidates_path = str(output_root / "aggregates" / "candidates.parquet").replace("\\", "/")
        agg_runs_path = str(output_root / "aggregates" / "runs.parquet").replace("\\", "/")
        self.add_node(DAGNode(
            node_id=agg_node_id,
            node_type=NodeType.AGGREGATION,
            description="Aggregate candidate-level and run-level records into unified datasets",
            dependencies=target_run_node_ids,
            expected_output_path=agg_runs_path,
        ))

        # 6. Statistical Analysis Node
        stats_node_id = "node_statistics"
        stats_csv_path = str(output_root / "statistics" / "effects.csv").replace("\\", "/")
        self.add_node(DAGNode(
            node_id=stats_node_id,
            node_type=NodeType.STATISTICAL_ANALYSIS,
            description="Compute seed-paired bootstrap CIs, censored time-to-threshold, and multiplicity corrections",
            dependencies={agg_node_id},
            expected_output_path=stats_csv_path,
        ))

        # 7. Blinded QE Audit Selection Node
        qe_sel_node_id = "node_qe_selection"
        qe_sel_path = str(output_root / "qe_audit" / "selection.json").replace("\\", "/")
        self.add_node(DAGNode(
            node_id=qe_sel_node_id,
            node_type=NodeType.QE_AUDIT_SELECTION,
            description="Deterministic blinded selection of 10 audit candidates across conditions and targets",
            dependencies={agg_node_id},
            payload={"candidate_count": spec.qe_audit_config.candidate_count},
            expected_output_path=qe_sel_path,
        ))

        # 8. QE Audit Execution Node
        qe_exec_node_id = "node_qe_execution"
        qe_results_path = str(output_root / "qe_audit" / "results.csv").replace("\\", "/")
        self.add_node(DAGNode(
            node_id=qe_exec_node_id,
            node_type=NodeType.QE_AUDIT_EXECUTION,
            description="Execute Quantum ESPRESSO relaxations for selected candidates and required decomposition phases",
            dependencies={qe_sel_node_id},
            payload={"config": asdict(spec.qe_audit_config) if hasattr(spec.qe_audit_config, "__dataclass_fields__") else spec.qe_audit_config},
            expected_output_path=qe_results_path,
        ))

        # 9. Report & Paper-Facing Artifacts Node
        report_node_id = "node_report_generation"
        claim_matrix_path = str(output_root / "paper" / "claim_evidence_matrix.md").replace("\\", "/")
        self.add_node(DAGNode(
            node_id=report_node_id,
            node_type=NodeType.REPORT_GENERATION,
            description="Generate publication-ready figures, tables, claim matrix, and reproducibility checklist",
            dependencies={stats_node_id, qe_exec_node_id},
            expected_output_path=claim_matrix_path,
        ))

    def validate(self) -> None:
        """Validate DAG: check for cycles, missing parents, and expected run counts."""
        # 1. Check dependencies exist
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise DAGValidationError(f"Node '{node_id}' depends on non-existent node '{dep}'")

        # 2. Cycle detection via Kahn's algorithm / topological sort
        in_degree = {nid: len(n.dependencies) for nid, n in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        # Build adjacency
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                adjacency[dep].append(nid)

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adjacency[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(self.nodes):
            raise DAGValidationError("Cycle detected in experiment DAG")

        # 3. Exact run count calculation
        expected_target_runs = len(self.spec.target_tasks) * len(self.spec.conditions) * len(self.spec.master_seeds)
        actual_target_nodes = sum(1 for n in self.nodes.values() if n.node_type == NodeType.TARGET_CAMPAIGN_RUN)
        if actual_target_nodes != expected_target_runs:
            raise DAGValidationError(
                f"Target run count mismatch: expected {expected_target_runs}, found {actual_target_nodes}"
            )

        expected_source_runs = len(self.spec.master_seeds)
        actual_source_nodes = sum(1 for n in self.nodes.values() if n.node_type == NodeType.SOURCE_MEMORY_RUN)
        if actual_source_nodes != expected_source_runs:
            raise DAGValidationError(
                f"Source run count mismatch: expected {expected_source_runs}, found {actual_source_nodes}"
            )

        # Every memory target must be paired with exactly one immutable source
        # snapshot node for the same seed.  This catches accidental omission of
        # a dependency even when topological sorting still succeeds.
        for node in self.nodes.values():
            if node.node_type == NodeType.TARGET_CAMPAIGN_RUN and node.payload.get("condition") in MEMORY_ARMS:
                seed = node.payload.get("seed")
                expected = f"node_snapshot_source_seed{seed}"
                if expected not in node.dependencies:
                    raise DAGValidationError(f"Memory target '{node.node_id}' is not dependent on '{expected}'")

    def topological_order(self) -> List[DAGNode]:
        """Return DAG nodes in a deterministic valid topological execution order."""
        in_degree = {nid: len(n.dependencies) for nid, n in self.nodes.items()}
        queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[DAGNode] = []

        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.dependencies:
                adjacency[dep].append(nid)

        while queue:
            curr = queue.pop(0)
            order.append(self.nodes[curr])
            for neighbor in sorted(adjacency[curr]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort()

        return order

    @property
    def total_run_count(self) -> int:
        return sum(
            1 for n in self.nodes.values()
            if n.node_type in (NodeType.SOURCE_MEMORY_RUN, NodeType.TARGET_CAMPAIGN_RUN)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.spec.experiment_id,
            "spec_hash": self.spec.spec_hash,
            "total_nodes": len(self.nodes),
            "total_runs": self.total_run_count,
            "nodes": {nid: n.to_dict() for nid, n in sorted(self.nodes.items())},
        }
