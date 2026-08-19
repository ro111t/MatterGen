"""
Streamlit UI for the Mattergen Autonomous Materials Discovery Agent.

A thin wrapper around MaterialsDiscoveryCampaign that lets you configure,
run, and inspect discovery campaigns from a browser.

Run with:
    streamlit run ui/app.py
"""

import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# Make project root importable when running `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.generator import HAS_MATTERGEN


st.set_page_config(
    page_title="AMDA | Autonomous Materials Discovery Agent",
    page_icon="�",
    layout="wide",
)

# --- Custom sci-fi / lab tech theme -------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .amda-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 0 24px rgba(6, 182, 212, 0.15);
        }

        .amda-logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            color: #22d3ee;
            text-shadow: 0 0 12px rgba(34, 211, 238, 0.5);
            margin: 0;
        }

        .amda-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .amda-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-top: 0.75rem;
        }

        .amda-badge-ok {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.35);
        }

        .amda-badge-warn {
            background: rgba(251, 146, 60, 0.15);
            color: #fb923c;
            border: 1px solid rgba(251, 146, 60, 0.35);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem;
        }

        div[data-testid="stMetric"] > div:first-child {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetric"] > div:nth-child(2) {
            color: #22d3ee !important;
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.6rem !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #1e293b;
            border-radius: 8px 8px 0 0;
            border: 1px solid #334155;
            border-bottom: none;
            color: #94a3b8;
        }

        .stTabs [aria-selected="true"] {
            background: #0f172a !important;
            color: #22d3ee !important;
            border-top: 2px solid #22d3ee;
        }

        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: linear-gradient(90deg, #0891b2 0%, #22d3ee 100%);
            color: #0f172a;
            border: none;
        }

        .stButton>button:hover {
            box-shadow: 0 0 16px rgba(34, 211, 238, 0.4);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

mg_available = bool(HAS_MATTERGEN)
badge_class = "amda-badge-ok" if mg_available else "amda-badge-warn"
badge_text = "MatterGen online" if mg_available else "MatterGen offline — mock mode"

st.markdown(
    f"""
    <div class="amda-header">
        <div class="amda-logo">AMDA</div>
        <div class="amda-subtitle">Autonomous Materials Discovery Agent — closed-loop generative materials research</div>
        <span class="amda-badge {badge_class}">{badge_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: lightweight status / help
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-family: 'Orbitron', sans-serif; color: #22d3ee; font-size: 1.4rem; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
            AMDA
        </div>
        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 1rem;">
            Autonomous Materials Discovery Agent
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Quick tips**")
    st.markdown(
        """
        - Start with **mock generation** for fast iteration.
        - Enable **DFT validation** to exercise the full pipeline.
        - Toggle **MatterGen** only when running in a Python 3.10 environment.
        """
    )
    if not HAS_MATTERGEN:
        st.warning("MatterGen is not importable; mock/pymatgen generation will be used.")


# ---------------------------------------------------------------------------
# Main panel: campaign configuration
# ---------------------------------------------------------------------------
st.markdown("### Campaign Configuration")

with st.expander("Identity & Objectives", expanded=True):
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        campaign_name = st.text_input("Campaign name", value="solid_electrolyte_discovery")
    with id_col2:
        domain = st.text_input("Domain", value="li_solid_electrolyte")

    obj_col1, obj_col2, obj_col3 = st.columns(3)
    with obj_col1:
        target_stability = st.number_input(
            "Target stability (eV/atom)", value=-0.1, step=0.05, format="%.3f"
        )
    with obj_col2:
        target_formation_energy = st.number_input(
            "Target formation energy (eV/atom)", value=-2.0, step=0.1, format="%.2f"
        )
    with obj_col3:
        use_career_memory = st.checkbox("Use career memory", value=True)

with st.expander("Constraints", expanded=True):
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        elements = st.text_input("Elements (comma-separated)", value="Li, P, S, O, Cl")
    with c_col2:
        max_atoms = st.number_input("Max atoms", value=20, step=1)

with st.expander("Run Settings", expanded=True):
    rs_col1, rs_col2, rs_col3 = st.columns(3)
    with rs_col1:
        iterations = st.number_input("Iterations", value=2, min_value=1, step=1)
    with rs_col2:
        candidates_per_iteration = st.number_input(
            "Candidates per iteration", value=15, min_value=1, step=1
        )
    with rs_col3:
        validation_top_k = st.number_input(
            "Top candidates to validate", value=3, min_value=1, step=1,
        )

with st.expander("Pipeline Stages", expanded=True):
    ps_col1, ps_col2 = st.columns(2)
    with ps_col1:
        use_validation = st.checkbox("Run DFT validation (mock)", value=True)
    with ps_col2:
        use_synthesis = st.checkbox(
            "Run synthesis feasibility",
            value=True,
            disabled=not use_validation,
        )

with st.expander("Generation Backend", expanded=HAS_MATTERGEN):
    if not HAS_MATTERGEN:
        st.warning(
            "MatterGen is not importable in this environment. "
            "Enable it below only for config export; generation will fall back to mock."
        )
    gb_col1, gb_col2, gb_col3 = st.columns([1, 2, 2])
    with gb_col1:
        use_mattergen = st.checkbox("Use MatterGen", value=False)
    with gb_col2:
        mattergen_pretrained = st.selectbox(
            "Pretrained model",
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
    with gb_col3:
        mattergen_sampling_config_name = st.selectbox(
            "Sampling config",
            options=["default", "csp"],
            index=0,
            disabled=not use_mattergen,
            help="Use 'csp' only when conditioning on target compositions with a CSP-trained model.",
        )
    gb_col4, gb_col5 = st.columns(2)
    with gb_col4:
        mattergen_batch_size = st.number_input(
            "Batch size", value=16, min_value=1, step=1, disabled=not use_mattergen
        )
    with gb_col5:
        mattergen_model_path = st.text_input(
            "Local checkpoint path (optional)", value="", disabled=not use_mattergen
        )

run_button = st.button("▶️ Run Campaign", use_container_width=True)


# ---------------------------------------------------------------------------
# Main panel: run campaign and show results
# ---------------------------------------------------------------------------
def _run_campaign_ui(
    campaign_name: str,
    domain: str,
    target_stability: float,
    target_formation_energy: float,
    element_list: list,
    max_atoms: int,
    iterations: int,
    candidates_per_iteration: int,
    use_career_memory: bool,
    use_validation: bool,
    validation_top_k: int,
    use_synthesis: bool,
    use_mattergen: bool,
    mattergen_pretrained: str,
    mattergen_sampling_config_name: str,
    mattergen_batch_size: int,
    mattergen_model_path: str,
):
    """Lazy import heavy deps and run one campaign."""
    from campaign import CampaignConfig, MaterialsDiscoveryCampaign
    from agents.orchestrator import CampaignObjective

    objective = CampaignObjective(
        target_properties={
            "stability": target_stability,
            "formation_energy": target_formation_energy,
        },
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
        verbose=False,  # streamlit captures its own output
        use_validation=use_validation,
        validation_top_k=validation_top_k,
        use_synthesis=use_synthesis,
        num_candidates=candidates_per_iteration,
        use_mattergen=use_mattergen,
        mattergen_pretrained=mattergen_pretrained,
        mattergen_sampling_config_name=mattergen_sampling_config_name,
        mattergen_batch_size=mattergen_batch_size,
        mattergen_model_path=mattergen_model_path if mattergen_model_path else None,
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
            target_stability=target_stability,
            target_formation_energy=target_formation_energy,
            element_list=element_list,
            max_atoms=max_atoms,
            iterations=iterations,
            candidates_per_iteration=candidates_per_iteration,
            use_career_memory=use_career_memory,
            use_validation=use_validation,
            validation_top_k=validation_top_k,
            use_synthesis=use_synthesis,
            use_mattergen=use_mattergen,
            mattergen_pretrained=mattergen_pretrained,
            mattergen_sampling_config_name=mattergen_sampling_config_name,
            mattergen_batch_size=mattergen_batch_size,
            mattergen_model_path=mattergen_model_path,
        )

    progress.progress(100, text="Campaign complete")
    st.success(f"Campaign finished: {results['campaign_name']}")

    # Export results
    export_col1, export_col2 = st.columns(2)
    report_json = json.dumps(results, indent=2, default=str)
    export_col1.download_button(
        label="📥 Download Report (JSON)",
        data=report_json,
        file_name=f"{results.get('campaign_name', 'campaign')}_report.json",
        mime="application/json",
        use_container_width=True,
    )
    if results.get("top_candidates"):
        csv_buffer = BytesIO()
        pd.DataFrame(results["top_candidates"]).to_csv(csv_buffer, index=False)
        export_col2.download_button(
            label="📥 Download Top Candidates (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"{results.get('campaign_name', 'campaign')}_top_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    backend = results.get("generation_backend", "unknown")
    pretrained = results.get("mattergen_pretrained")
    if backend == "mattergen" and pretrained:
        st.info(f"Generation backend: **MatterGen ({pretrained})**")
    else:
        st.info(f"Generation backend: **pymatgen mock**")

    # Summary metrics
    st.markdown("### Campaign Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Iterations", results.get("iterations", 0))
    mcol2.metric("Generated", results.get("total_generated", 0))
    mcol3.metric("Passed Screening", results.get("total_passed_screening", 0))
    mcol4.metric("Best Score", f"{results.get('best_score_ever', 0):.3f}")

    mcol5, mcol6, mcol7, mcol8 = st.columns(4)
    mcol5.metric("Validated", results.get("total_validated", 0))
    mcol6.metric("Converged", results.get("total_converged", 0))
    mcol7.metric("Synthesis Feasible", results.get("total_synthesis_feasible", 0))
    mcol8.metric("Best Validated Stability", f"{results.get('best_validated_stability_ever', 0):.3f}")

    # Load the latest checkpoint for per-iteration details
    output_dir = Path(f"./campaigns/{domain}")
    checkpoint_files = sorted(output_dir.glob("checkpoint_*.json"))
    latest_checkpoint = None
    if checkpoint_files:
        with open(checkpoint_files[-1]) as f:
            latest_checkpoint = json.load(f)

    tab_summary, tab_candidates, tab_analysis, tab_checkpoints = st.tabs([
        "Summary", "Candidates", "Analysis", "Iteration History"
    ])

    with tab_summary:
        st.subheader("Final Report")
        st.json(results)

        top_candidates = results.get("top_candidates", [])
        if top_candidates:
            st.subheader("Top Candidates from Career Memory")
            st.dataframe(pd.DataFrame(top_candidates), use_container_width=True)

    with tab_candidates:
        st.subheader("Latest Iteration Screened Candidates")
        if latest_checkpoint and latest_checkpoint.get("results_history"):
            last_iter = latest_checkpoint["results_history"][-1]
            screened = last_iter.get("screened_candidates", [])
            if screened:
                screened_df = pd.DataFrame(screened)
                st.dataframe(screened_df, use_container_width=True)

                st.markdown("---")
                st.markdown("**Score component breakdown (top 10 candidates)**")
                component_cols = ["stability", "relaxation_quality", "target_property_match", "composition_novelty"]
                top_n = screened_df.sort_values("score", ascending=False).head(10)
                components_df = top_n["score_components"].apply(pd.Series)[component_cols]
                components_df.index = top_n["structure_id"].values
                st.bar_chart(components_df)

                st.markdown("**Pass / fail summary**")
                pass_counts = screened_df["passes_filters"].value_counts().rename({True: "Pass", False: "Fail"})
                st.bar_chart(pass_counts)

                st.markdown("**Filter rejection reasons**")
                reasons = []
                for reason_list in screened_df["filter_reasons"]:
                    reasons.extend(reason_list)
                if reasons:
                    reason_counts = pd.Series(reasons).value_counts()
                    st.bar_chart(reason_counts)
                else:
                    st.info("No candidates rejected by filters.")
            else:
                st.info("No screened candidate details in the latest checkpoint.")
        else:
            st.info("Run a campaign to inspect screened candidates.")

    with tab_analysis:
        if latest_checkpoint and latest_checkpoint.get("results_history"):
            last_iter = latest_checkpoint["results_history"][-1]
            analysis = last_iter.get("analysis", {})
            st.subheader("Latest Iteration Backend")
            st.write(f"Generation backend: **{last_iter.get('generation_backend', 'unknown')}**")

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
                        "best_validated_stability": insights.get("best_validated_stability"),
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
