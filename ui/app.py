"""
Streamlit UI for the Mattergen Autonomous Materials Discovery Agent.

A thin wrapper around MaterialsDiscoveryCampaign that lets you configure,
run, and inspect discovery campaigns from a browser.

Run with:
    streamlit run ui/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable when running `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Mattergen Discovery Agent",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Mattergen Autonomous Materials Discovery Agent")
st.markdown(
    "Configure a discovery campaign, run it, and inspect the results. "
    "This UI wraps the existing `MaterialsDiscoveryCampaign` pipeline."
)

# ---------------------------------------------------------------------------
# Sidebar: campaign configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Campaign Configuration")

    campaign_name = st.text_input("Campaign name", value="solid_electrolyte_discovery")
    domain = st.text_input("Domain", value="li_solid_electrolyte")

    st.subheader("Target Properties")
    st.info("Hull metrics require an offline-built, certified frozen reference set evaluated with the same CHGNet checkpoint.")
    thermodynamics_reference_set_path = st.text_input(
        "Frozen reference-set JSON (optional)", value="",
        help="No reference data is downloaded at runtime; the adjacent .sha256 file is required.",
    )
    thermodynamics_retain_threshold_ev_per_atom = st.number_input(
        "Retain hull threshold (eV/atom)", value=0.10, min_value=0.0, step=0.01,
        format="%.2f", help="Candidates at or below this predicted energy above hull are retained.",
    )
    thermodynamics_stable_threshold_ev_per_atom = st.number_input(
        "Stable hull threshold (eV/atom)", value=0.03, min_value=0.0, step=0.01,
        format="%.2f", help="Candidates at or below this predicted energy above hull are labeled predicted stable.",
    )

    st.subheader("Constraints")
    elements = st.text_input(
        "Elements (comma-separated)", value="Li, P, S, O, Cl"
    )
    max_atoms = st.number_input("Max atoms", value=20, step=1)

    st.subheader("Run Settings")
    iterations = st.number_input("Iterations", value=2, min_value=1, step=1)
    candidates_per_iteration = st.number_input(
        "Candidates per iteration", value=15, min_value=1, step=1
    )
    proposal_budget = st.number_input(
        "Proposal budget (paper default 400)", value=400, min_value=1, step=1,
        help="Every generated candidate consumes one proposal slot.",
    )
    oracle_budget = st.number_input(
        "Oracle budget (paper default 200)", value=200, min_value=1, step=1,
        help="Every geometrically valid candidate submitted to CHGNet consumes one oracle slot, including cache hits.",
    )
    geometry_min_distance = st.number_input(
        "Minimum periodic distance (Å)", value=0.8, min_value=0.001, step=0.1,
    )
    use_career_memory = st.checkbox("Use career memory", value=True)
    memory_mode = st.selectbox(
        "Career memory view",
        options=["none", "text_summary", "structured_provenance", "shuffled_control"],
        index=2,
        disabled=not use_career_memory,
        help="Shuffled control preserves marginals but is invalid for scientific decision support.",
    )
    memory_seed = st.number_input("Career memory seed", value=0, step=1, disabled=not use_career_memory)
    run_mode = st.selectbox(
        "Run mode", options=["development", "research"], index=0,
        help="Research mode fails closed unless MatterGen, CHGNet, scientific validation, and required thermodynamics are configured.",
    )

    st.subheader("Pipeline Stages")
    use_validation = st.checkbox("Run DFT validation (development mock)", value=True)
    validation_top_k = st.number_input(
        "Top candidates to validate", value=3, min_value=1, step=1,
        disabled=not use_validation,
    )
    use_synthesis = st.checkbox("Run synthesis feasibility (development-only)", value=True, disabled=not use_validation or run_mode == "research")
    if run_mode == "research":
        use_synthesis = False

    st.subheader("Generation Backend")
    use_mattergen = st.checkbox("Use MatterGen", value=False)
    mattergen_pretrained = st.selectbox(
        "MatterGen pretrained model",
        options=[
            "mattergen_base",
            "chemical_system",
            "space_group",
            "dft_band_gap",
            "dft_mag_density",
            "ml_bulk_modulus",
            "dft_mag_density_hhi_score",
            "chemical_system_energy_above_hull",
            "mp_20_base",
        ],
        index=0,
        disabled=not use_mattergen,
    )
    mattergen_sampling_config_name = st.selectbox(
        "MatterGen sampling config",
        options=["default", "csp"],
        index=0,
        disabled=not use_mattergen,
        help="Use 'csp' only when conditioning on target compositions with a CSP-trained model.",
    )
    mattergen_batch_size = st.number_input(
        "MatterGen batch size", value=16, min_value=1, step=1, disabled=not use_mattergen
    )
    mattergen_model_path = st.text_input(
        "Local MatterGen checkpoint path (optional)", value="", disabled=not use_mattergen
    )

    run_button = st.button("▶️ Run Campaign", use_container_width=True)


# ---------------------------------------------------------------------------
# Main panel: run campaign and show results
# ---------------------------------------------------------------------------
def _run_campaign_ui(
    campaign_name: str,
    domain: str,
    element_list: list,
    max_atoms: int,
    iterations: int,
    candidates_per_iteration: int,
    proposal_budget: int,
    oracle_budget: int,
    geometry_min_distance: float,
    use_career_memory: bool,
    memory_mode: str,
    memory_seed: int,
    run_mode: str,
    use_validation: bool,
    validation_top_k: int,
    use_synthesis: bool,
    use_mattergen: bool,
    mattergen_pretrained: str,
    mattergen_sampling_config_name: str,
    mattergen_batch_size: int,
    mattergen_model_path: str,
    thermodynamics_reference_set_path: str,
    thermodynamics_retain_threshold_ev_per_atom: float,
    thermodynamics_stable_threshold_ev_per_atom: float,
):
    """Lazy import heavy deps and run one campaign."""
    from campaign import CampaignConfig, MaterialsDiscoveryCampaign
    from agents.orchestrator import CampaignObjective

    objective = CampaignObjective(
        target_properties={},
        constraints={"elements": element_list, "max_atoms": max_atoms},
        success_criteria={"min_score": 999.0},  # run full iteration budget
        domain=domain,
        max_iterations=iterations,
    )

    config = CampaignConfig(
        name=campaign_name,
        objective=objective,
        output_dir=Path(f"./campaigns/{domain}"),
        use_career_memory=use_career_memory,
        memory_mode=memory_mode,
        memory_seed=memory_seed,
        run_mode=run_mode,
        verbose=False,  # streamlit captures its own output
        use_validation=use_validation,
        validation_top_k=validation_top_k,
        use_synthesis=use_synthesis,
        num_candidates=candidates_per_iteration,
        proposal_budget=proposal_budget,
        oracle_budget=oracle_budget,
        geometry_min_distance=geometry_min_distance,
        use_mattergen=use_mattergen,
        mattergen_pretrained=mattergen_pretrained,
        mattergen_sampling_config_name=mattergen_sampling_config_name,
        mattergen_batch_size=mattergen_batch_size,
        mattergen_model_path=mattergen_model_path if mattergen_model_path else None,
        thermodynamics_reference_set_path=(thermodynamics_reference_set_path or None),
        thermodynamics_retain_threshold_ev_per_atom=thermodynamics_retain_threshold_ev_per_atom,
        thermodynamics_stable_threshold_ev_per_atom=thermodynamics_stable_threshold_ev_per_atom,
    )

    campaign = MaterialsDiscoveryCampaign(config)
    return campaign.run_campaign()


if run_button:
    element_list = [e.strip() for e in elements.split(",") if e.strip()]

    progress = st.progress(0, text="Starting campaign...")

    with st.spinner("Running discovery campaign..."):
        results = _run_campaign_ui(
            campaign_name=campaign_name,
            domain=domain,
            element_list=element_list,
            max_atoms=max_atoms,
            iterations=iterations,
            candidates_per_iteration=candidates_per_iteration,
            proposal_budget=proposal_budget,
            oracle_budget=oracle_budget,
            geometry_min_distance=geometry_min_distance,
            use_career_memory=use_career_memory,
            memory_mode=memory_mode,
            memory_seed=memory_seed,
            run_mode=run_mode,
            use_validation=use_validation,
            validation_top_k=validation_top_k,
            use_synthesis=use_synthesis,
            use_mattergen=use_mattergen,
            mattergen_pretrained=mattergen_pretrained,
            mattergen_sampling_config_name=mattergen_sampling_config_name,
            mattergen_batch_size=mattergen_batch_size,
            mattergen_model_path=mattergen_model_path,
            thermodynamics_reference_set_path=thermodynamics_reference_set_path,
            thermodynamics_retain_threshold_ev_per_atom=thermodynamics_retain_threshold_ev_per_atom,
            thermodynamics_stable_threshold_ev_per_atom=thermodynamics_stable_threshold_ev_per_atom,
        )

    progress.progress(100, text="Campaign complete")
    st.success(f"Campaign finished: {results['campaign_name']}")

    backend = results.get("generation_backend", "unknown")
    pretrained = results.get("mattergen_pretrained")
    if backend == "mattergen" and pretrained:
        st.info(f"Generation backend: **MatterGen ({pretrained})**")
    else:
        st.info(f"Generation backend: **pymatgen mock**")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Iterations", results.get("iterations", 0))
    col2.metric("Generated", results.get("total_generated", 0))
    col3.metric("Passed Screening", results.get("total_passed_screening", 0))
    col4.metric("Best Score", f"{results.get('best_score_ever', 0):.3f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Validated", results.get("total_validated", 0))
    col6.metric("Converged", results.get("total_converged", 0))
    col7.metric("Synthesis Feasible", results.get("total_synthesis_feasible", 0))
    col8.metric("Thermodynamics", "pending hull oracle")

    st.subheader("Geometry and Oracle Budget")
    budget_cols = st.columns(6)
    budget_cols[0].metric("Proposals", results.get("proposals_generated", results.get("total_generated", 0)))
    budget_cols[1].metric("Geometry valid", results.get("geometry_valid", 0))
    budget_cols[2].metric("Invalid geometry", results.get("invalid_geometry", 0))
    budget_cols[3].metric("Oracle evaluations", results.get("oracle_evaluations", 0))
    budget_cols[4].metric("Proposal remaining", results.get("proposal_budget_remaining", "unlimited"))
    budget_cols[5].metric("Oracle remaining", results.get("oracle_budget_remaining", "unlimited"))
    if results.get("termination_reason"):
        st.caption(f"Termination: {results['termination_reason']}")

    # Load the latest checkpoint for per-iteration details
    output_dir = Path(f"./campaigns/{domain}")
    checkpoint_files = sorted(output_dir.glob("checkpoint_*.json"))
    latest_checkpoint = None
    if checkpoint_files:
        with open(checkpoint_files[-1]) as f:
            latest_checkpoint = json.load(f)

    tab_summary, tab_analysis, tab_checkpoints = st.tabs([
        "Summary", "Analysis", "Iteration History"
    ])

    with tab_summary:
        st.subheader("Final Report")
        st.json(results)

        top_candidates = results.get("top_candidates", [])
        if top_candidates:
            st.subheader("Top Candidates from Career Memory")
            st.dataframe(pd.DataFrame(top_candidates), use_container_width=True)

    with tab_analysis:
        if latest_checkpoint and latest_checkpoint.get("results_history"):
            last_iter = latest_checkpoint["results_history"][-1]
            analysis = last_iter.get("analysis", {})

            st.subheader("Latest Iteration Analysis")
            st.markdown(
                f"**Top candidate:** `{analysis.get('top_candidate_id')}`  "
                f"score={analysis.get('top_candidate_score', 0):.3f}"
            )

            if analysis.get("ml_vs_dft_mae"):
                metrics_df = pd.DataFrame({
                    "MAE": analysis.get("ml_vs_dft_mae", {}),
                    "RMSE": analysis.get("ml_vs_dft_rmse", {}),
                    "Bias": analysis.get("ml_vs_dft_bias", {}),
                    "Pearson r": analysis.get("pearson_r", {}),
                }).T
                st.markdown("**ML vs DFT Calibration**")
                st.dataframe(metrics_df, use_container_width=True)

            if analysis.get("failure_modes"):
                st.markdown("**Failure Modes**")
                for mode in analysis["failure_modes"]:
                    st.markdown(f"- {mode}")

            if analysis.get("insights"):
                st.markdown("**Insights**")
                for insight in analysis["insights"]:
                    st.markdown(f"- {insight}")

            # Validation results table
            validation_results = last_iter.get("validation_results", [])
            if validation_results:
                st.markdown("**Validation Results**")
                rows = []
                for v in validation_results:
                    rows.append({
                        "structure_id": v.get("structure_id"),
                        "calculator": v.get("calculator"),
                        "converged": v.get("converged"),
                        "cost_hours": v.get("cost_hours"),
                        **v.get("properties", {}),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Synthesis results table
            synthesis_results = last_iter.get("synthesis_results", [])
            if synthesis_results:
                st.markdown("**Synthesis Feasibility**")
                rows = []
                for s in synthesis_results:
                    rows.append({
                        "structure_id": s.get("structure_id"),
                        "feasible": s.get("feasible"),
                        "feasibility": s.get("feasibility_score"),
                        "difficulty": s.get("difficulty_score"),
                        "route": s.get("synthesis_route"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No detailed analysis available. Run the campaign with validation enabled.")

    with tab_checkpoints:
        if checkpoint_files:
            history = []
            for cp in checkpoint_files:
                with open(cp) as f:
                    data = json.load(f)
                for r in data.get("results_history", []):
                    insights = r.get("insights", {})
                    history.append({
                        "iteration": r.get("iteration"),
                        "generated": r.get("num_generated"),
                        "passed": insights.get("num_passed"),
                        "validated": insights.get("num_validated"),
                        "converged": insights.get("num_converged"),
                        "synthesis_feasible": insights.get("num_synthesis_feasible"),
                        "best_score": insights.get("best_score"),
                        "thermodynamics_available": insights.get("thermodynamics_metrics_available", False),
                    })
            if history:
                hist_df = pd.DataFrame(history).drop_duplicates("iteration")
                st.dataframe(hist_df, use_container_width=True)
                st.line_chart(
                    hist_df.set_index("iteration")[[
                        "best_score", "passed", "validated", "synthesis_feasible"
                    ]]
                )
        else:
            st.info("No checkpoints found yet.")

else:
    st.info("Configure a campaign in the sidebar and click **Run Campaign** to start.")
