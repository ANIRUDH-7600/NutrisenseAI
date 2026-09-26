"""Comprehensive Exploratory Data Analysis (EDA) Module for NutriSense AI.

Analyzes demographic, maternal, socioeconomic, environmental, and morbidity patterns
associated with childhood stunting, underweight, and wasting across the NFHS-5 cohort.
Generates publication-quality figures and structured analytical summaries.
"""
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_DATA_PATH = "data/interim/cleaned_u5_with_targets.csv.gz"
OUTPUT_DIR_FIGURES = "reports/figures"
OUTPUT_SUMMARY_JSON = "data/interim/eda_summary.json"

# State mapping dictionary from official NFHS-5 codebook (v024)
STATE_NAMES = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "Delhi", 8: "Rajasthan", 9: "Uttar Pradesh",
    10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh", 13: "Nagaland", 14: "Manipur",
    15: "Mizoram", 16: "Tripura", 17: "Meghalaya", 18: "Assam", 19: "West Bengal",
    20: "Jharkhand", 21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh",
    24: "Gujarat", 25: "Daman & Diu", 26: "Dadra & Nagar Haveli", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep", 32: "Kerala",
    33: "Tamil Nadu", 34: "Puducherry", 35: "Andaman & Nicobar", 36: "Telangana", 37: "Ladakh"
}

def cramers_v(contingency_table):
    """Calculate Cramer's V statistic for categorical-categorical association."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom == 0:
        return 0.0
    return np.sqrt(phi2corr / denom)

def run_eda():
    print(f"[*] Step 6: Loading target-augmented dataset from {INPUT_DATA_PATH}...")
    df = pd.read_csv(INPUT_DATA_PATH)
    total_records = len(df)
    print(f"    Loaded {total_records:,} records.")
    os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. TARGET DISTRIBUTIONS & DESCRIPTIVE ESTIMATES USING DHS SAMPLE WEIGHT
    # -------------------------------------------------------------------------
    def target_metrics(target_col):
        s = df[target_col].dropna()
        w = df.loc[df[target_col].notna(), "sample_weight"]
        unweighted_prev = (s == 1.0).mean() * 100
        weighted_prev = np.average(s == 1.0, weights=w) * 100
        return {
            "valid_n": int(len(s)),
            "positive_n": int((s == 1.0).sum()),
            "negative_n": int((s == 0.0).sum()),
            "unweighted_prevalence_pct": round(unweighted_prev, 2),
            "weighted_prevalence_pct": round(weighted_prev, 2),
            "class_ratio_neg_to_pos": round((s == 0.0).sum() / (s == 1.0).sum(), 2)
        }

    target_summary = {
        "stunting": target_metrics("stunting"),
        "underweight": target_metrics("underweight"),
        "wasting": target_metrics("wasting"),
        "stunting_severe": target_metrics("stunting_severe"),
        "underweight_severe": target_metrics("underweight_severe"),
        "wasting_severe": target_metrics("wasting_severe")
    }

    # -------------------------------------------------------------------------
    # 2. BIVARIATE PREVALENCE STRATIFICATIONS
    # -------------------------------------------------------------------------
    # Wealth index (v190: 1=Poorest to 5=Richest)
    wealth_labels = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
    df["wealth_label"] = df["v190"].map(wealth_labels)
    
    # Maternal education (v106: 0=No edu, 1=Primary, 2=Secondary, 3=Higher)
    edu_labels = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
    df["edu_label"] = df["v106"].map(edu_labels)

    # Place of residence (v025: 1=Urban, 2=Rural)
    res_labels = {1: "Urban", 2: "Rural"}
    df["residence_label"] = df["v025"].map(res_labels)

    # Maternal BMI category (v445_clean)
    def categorize_bmi(bmi):
        if pd.isna(bmi):
            return np.nan
        if bmi < 18.5:
            return "Underweight (<18.5)"
        elif bmi < 25.0:
            return "Normal (18.5-24.9)"
        elif bmi < 30.0:
            return "Overweight (25.0-29.9)"
        else:
            return "Obese (>=30.0)"
    df["maternal_bmi_cat"] = df["v445_clean"].apply(categorize_bmi)

    # Child age groups
    def age_group(m):
        if m < 6:
            return "0-5 mo"
        elif m < 12:
            return "6-11 mo"
        elif m < 24:
            return "12-23 mo"
        elif m < 36:
            return "24-35 mo"
        elif m < 48:
            return "36-47 mo"
        else:
            return "48-59 mo"
    df["age_group"] = df["hw1"].apply(age_group)
    age_order = ["0-5 mo", "6-11 mo", "12-23 mo", "24-35 mo", "36-47 mo", "48-59 mo"]

    def calc_group_prevalence(group_col, order=None):
        out = {}
        for g, sub in df.groupby(group_col, observed=False):
            if pd.isna(g):
                continue
            out[str(g)] = {
                "n_children": int(len(sub)),
                "stunting_prev_pct": round(sub["stunting"].mean() * 100, 2),
                "underweight_prev_pct": round(sub["underweight"].mean() * 100, 2),
                "wasting_prev_pct": round(sub["wasting"].mean() * 100, 2)
            }
        return out

    bivariate_summary = {
        "by_wealth_quintile": calc_group_prevalence("wealth_label"),
        "by_maternal_education": calc_group_prevalence("edu_label"),
        "by_residence": calc_group_prevalence("residence_label"),
        "by_maternal_bmi": calc_group_prevalence("maternal_bmi_cat"),
        "by_age_group": calc_group_prevalence("age_group")
    }

    # -------------------------------------------------------------------------
    # 3. STATISTICAL ASSOCIATIONS (CRAMER'S V & POINT BISERIAL)
    # -------------------------------------------------------------------------
    candidate_cat_vars = [
        ("wealth_label", "Wealth Quintile"),
        ("edu_label", "Maternal Education"),
        ("residence_label", "Place of Residence"),
        ("maternal_bmi_cat", "Maternal BMI Category"),
        ("b4", "Child Sex"),
        ("b0", "Twin / Multiple"),
        ("diarrhea_recent", "Recent Diarrhea"),
        ("fever_recent", "Recent Fever"),
        ("cough_recent", "Recent Cough / ARI"),
        ("still_breastfeeding", "Still Breastfeeding"),
        ("v119", "Household Electricity"),
        ("m18_clean", "Size at Birth")
    ]

    cramers_matrix = []
    for var_col, var_label in candidate_cat_vars:
        sub_valid = df[[var_col, "stunting", "underweight", "wasting"]].dropna()
        cv_stunt = cramers_v(pd.crosstab(sub_valid[var_col], sub_valid["stunting"]))
        cv_under = cramers_v(pd.crosstab(sub_valid[var_col], sub_valid["underweight"]))
        cv_waste = cramers_v(pd.crosstab(sub_valid[var_col], sub_valid["wasting"]))
        cramers_matrix.append({
            "feature": var_label,
            "stunting_cramers_v": round(float(cv_stunt), 4),
            "underweight_cramers_v": round(float(cv_under), 4),
            "wasting_cramers_v": round(float(cv_waste), 4)
        })

    # -------------------------------------------------------------------------
    # 4. GEOGRAPHIC VARIATIONS ACROSS INDIAN STATES (v024)
    # -------------------------------------------------------------------------
    df["state_name"] = df["v024"].map(STATE_NAMES)
    state_records = []
    for st_id, st_name in STATE_NAMES.items():
        st_sub = df[df["v024"] == st_id]
        if len(st_sub) == 0:
            continue
        s_prev = st_sub["stunting"].mean() * 100
        u_prev = st_sub["underweight"].mean() * 100
        w_prev = st_sub["wasting"].mean() * 100
        
        # Survey-weighted state prevalence
        sw = st_sub.loc[st_sub["stunting"].notna(), "sample_weight"]
        uw = st_sub.loc[st_sub["underweight"].notna(), "sample_weight"]
        ww = st_sub.loc[st_sub["wasting"].notna(), "sample_weight"]
        
        sw_prev = np.average(st_sub.loc[st_sub["stunting"].notna(), "stunting"] == 1.0, weights=sw) * 100 if len(sw) > 0 else s_prev
        uw_prev = np.average(st_sub.loc[st_sub["underweight"].notna(), "underweight"] == 1.0, weights=uw) * 100 if len(uw) > 0 else u_prev
        ww_prev = np.average(st_sub.loc[st_sub["wasting"].notna(), "wasting"] == 1.0, weights=ww) * 100 if len(ww) > 0 else w_prev

        state_records.append({
            "state_id": st_id,
            "state_name": st_name,
            "sample_size": int(len(st_sub)),
            "unweighted_stunting_pct": round(s_prev, 2),
            "weighted_stunting_pct": round(sw_prev, 2),
            "unweighted_wasting_pct": round(w_prev, 2),
            "weighted_wasting_pct": round(ww_prev, 2),
            "unweighted_underweight_pct": round(u_prev, 2),
            "weighted_underweight_pct": round(uw_prev, 2)
        })

    # -------------------------------------------------------------------------
    # 5. GENERATE PUBLICATION-QUALITY FIGURES
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    # FIGURE 1: Target Prevalence & Severity Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    targets = ["Stunting (HAZ)", "Underweight (WAZ)", "Wasting (WHZ)"]
    mod_prev = [target_summary["stunting"]["unweighted_prevalence_pct"],
                target_summary["underweight"]["unweighted_prevalence_pct"],
                target_summary["wasting"]["unweighted_prevalence_pct"]]
    sev_prev = [target_summary["stunting_severe"]["unweighted_prevalence_pct"],
                target_summary["underweight_severe"]["unweighted_prevalence_pct"],
                target_summary["wasting_severe"]["unweighted_prevalence_pct"]]
    x = np.arange(len(targets))
    width = 0.35
    rects1 = ax.bar(x - width/2, mod_prev, width, label="Moderate-or-Severe (< -2 SD)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, sev_prev, width, label="Severe (< -3 SD)", color="#c93b2b")
    ax.set_ylabel("Analytical Sample Prevalence (%)")
    ax.set_title("Figure 1: Childhood Undernutrition Target Prevalence in NFHS-5 Cohort", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend(frameon=True)
    ax.set_ylim(0, 45)
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR_FIGURES, "fig1_target_prevalence_and_imbalance.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # FIGURE 2: Predictor Missingness Profile
    with open("data/interim/cleaning_audit.json", "r") as f:
        audit_meta = json.load(f)
    miss_df = pd.DataFrame(audit_meta["missingness_audit"])
    # Filter to non-target variables with missingness > 0
    miss_pred = miss_df[
        (~miss_df["variable"].str.contains("hw7")) & 
        (miss_df["missing_pct"] > 0)
    ].sort_values("missing_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4a7bb0" if "Structural" not in cat else "#e67e22" for cat in miss_pred["missingness_category"]]
    bars = ax.barh(miss_pred["variable"], miss_pred["missing_pct"], color=colors)
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_title("Figure 2: Predictor Missingness Profile across Living Under-5 Cohort (N = 221,263)", fontsize=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 45)
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color="#e67e22", label="Structural Non-Applicability (Firstborn child)"),
        plt.Rectangle((0,0),1,1, color="#4a7bb0", label="Informational Uncertainty / Refused / Consent")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR_FIGURES, "fig2_missingness_profile.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # FIGURE 3: Age Dynamics across Malnutrition Targets
    age_prev_df = pd.DataFrame([
        {
            "Age Group": grp,
            "Stunting": bivariate_summary["by_age_group"][grp]["stunting_prev_pct"],
            "Underweight": bivariate_summary["by_age_group"][grp]["underweight_prev_pct"],
            "Wasting": bivariate_summary["by_age_group"][grp]["wasting_prev_pct"]
        } for grp in age_order
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(age_prev_df["Age Group"], age_prev_df["Stunting"], marker="o", linewidth=2.2, color="#2b5c8f", label="Stunting (HAZ < -2 SD)")
    ax.plot(age_prev_df["Age Group"], age_prev_df["Underweight"], marker="s", linewidth=2.2, color="#27ae60", label="Underweight (WAZ < -2 SD)")
    ax.plot(age_prev_df["Age Group"], age_prev_df["Wasting"], marker="^", linewidth=2.2, color="#e74c3c", label="Wasting (WHZ < -2 SD)")
    ax.set_ylabel("Analytical Sample Prevalence (%)")
    ax.set_xlabel("Child Completed Age Group (Months)")
    ax.set_title("Figure 3: Age Dynamics of Childhood Undernutrition in India (NFHS-5)", fontsize=12, pad=12)
    ax.legend(frameon=True)
    ax.set_ylim(10, 45)
    for col, color in [("Stunting", "#2b5c8f"), ("Underweight", "#27ae60"), ("Wasting", "#e74c3c")]:
        for x_idx, y_val in enumerate(age_prev_df[col]):
            ax.annotate(f"{y_val:.1f}%", (x_idx, y_val + 0.8), ha="center", fontsize=8, color=color, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR_FIGURES, "fig3_age_dynamics_by_outcome.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # FIGURE 4: Socioeconomic & Maternal Gradients (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 4A: Wealth Quintile
    w_order = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    w_data = [bivariate_summary["by_wealth_quintile"][w]["stunting_prev_pct"] for w in w_order]
    axes[0, 0].bar(w_order, w_data, color="#2980b9")
    axes[0, 0].set_title("A. Stunting Prevalence by Household Wealth Quintile", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Stunting Prevalence (%)")
    for idx, v in enumerate(w_data):
        axes[0, 0].text(idx, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)

    # 4B: Maternal Education
    e_order = ["No Education", "Primary", "Secondary", "Higher"]
    e_data = [bivariate_summary["by_maternal_education"][e]["stunting_prev_pct"] for e in e_order]
    axes[0, 1].bar(e_order, e_data, color="#16a085")
    axes[0, 1].set_title("B. Stunting Prevalence by Mother's Education Level", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Stunting Prevalence (%)")
    for idx, v in enumerate(e_data):
        axes[0, 1].text(idx, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)

    # 4C: Maternal BMI Category
    bmi_order = ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25.0-29.9)", "Obese (>=30.0)"]
    bmi_data = [bivariate_summary["by_maternal_bmi"][b]["underweight_prev_pct"] for b in bmi_order]
    axes[1, 0].bar(bmi_order, bmi_data, color="#8e44ad")
    axes[1, 0].set_title("C. Child Underweight Prevalence by Maternal BMI Category", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Child Underweight (%)")
    axes[1, 0].tick_params(axis="x", rotation=15)
    for idx, v in enumerate(bmi_data):
        axes[1, 0].text(idx, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)

    # 4D: Place of Residence
    res_order = ["Urban", "Rural"]
    r_stunt = [bivariate_summary["by_residence"][r]["stunting_prev_pct"] for r in res_order]
    r_waste = [bivariate_summary["by_residence"][r]["wasting_prev_pct"] for r in res_order]
    x_r = np.arange(2)
    axes[1, 1].bar(x_r - 0.15, r_stunt, 0.3, label="Stunting", color="#2b5c8f")
    axes[1, 1].bar(x_r + 0.15, r_waste, 0.3, label="Wasting", color="#c93b2b")
    axes[1, 1].set_title("D. Nutritional Deficit by Place of Residence", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x_r)
    axes[1, 1].set_xticklabels(res_order)
    axes[1, 1].set_ylabel("Prevalence (%)")
    axes[1, 1].legend()
    for idx, (st, ws) in enumerate(zip(r_stunt, r_waste)):
        axes[1, 1].text(idx - 0.15, st + 0.8, f"{st:.1f}%", ha="center", fontsize=9)
        axes[1, 1].text(idx + 0.15, ws + 0.8, f"{ws:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig4_path = os.path.join(OUTPUT_DIR_FIGURES, "fig4_socioeconomic_maternal_gradients.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # FIGURE 5: Morbidity Associations: Acute Wasting vs Chronic Stunting
    morb_items = [
        ("diarrhea_recent", "Recent Diarrhea (2 wks)"),
        ("fever_recent", "Recent Fever (2 wks)"),
        ("cough_recent", "Recent Cough / ARI (2 wks)")
    ]
    morb_plot_data = []
    for m_col, m_name in morb_items:
        yes_sub = df[df[m_col] == 1.0]
        no_sub = df[df[m_col] == 0.0]
        morb_plot_data.append({
            "Symptom": m_name,
            "Wasting (Symptom Yes)": round(yes_sub["wasting"].mean() * 100, 2),
            "Wasting (Symptom No)": round(no_sub["wasting"].mean() * 100, 2),
            "Stunting (Symptom Yes)": round(yes_sub["stunting"].mean() * 100, 2),
            "Stunting (Symptom No)": round(no_sub["stunting"].mean() * 100, 2)
        })
    morb_df = pd.DataFrame(morb_plot_data)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_m = np.arange(len(morb_df))
    w_m = 0.2
    ax.bar(x_m - 1.5*w_m, morb_df["Wasting (Symptom Yes)"], w_m, label="Wasting (Symptom Present)", color="#c0392b")
    ax.bar(x_m - 0.5*w_m, morb_df["Wasting (Symptom No)"], w_m, label="Wasting (Symptom Absent)", color="#f1948a")
    ax.bar(x_m + 0.5*w_m, morb_df["Stunting (Symptom Yes)"], w_m, label="Stunting (Symptom Present)", color="#2471a3")
    ax.bar(x_m + 1.5*w_m, morb_df["Stunting (Symptom No)"], w_m, label="Stunting (Symptom Absent)", color="#85c1e9")
    ax.set_xticks(x_m)
    ax.set_xticklabels(morb_df["Symptom"], fontsize=9)
    ax.set_ylabel("Outcome Prevalence (%)")
    ax.set_title("Figure 5: Bivariate Morbidity Symptom Associations with Wasting vs Stunting", fontsize=11, pad=12)
    ax.legend(frameon=True, fontsize=8)
    ax.set_ylim(0, 45)
    plt.tight_layout()
    fig5_path = os.path.join(OUTPUT_DIR_FIGURES, "fig5_morbidity_associations_acute_vs_chronic.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # FIGURE 6: Cramer's V Effect Size Heatmap
    cv_df = pd.DataFrame(cramers_matrix).set_index("feature")
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(cv_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={"label": "Cramer's V Association"}, ax=ax)
    ax.set_title("Figure 6: Categorical Association Matrix (Cramer's V with Malnutrition Targets)", fontsize=11, pad=12)
    ax.set_xlabel("Clinical Targets")
    ax.set_ylabel("Candidate Predictor Features")
    plt.tight_layout()
    fig6_path = os.path.join(OUTPUT_DIR_FIGURES, "fig6_statistical_association_cramers_v.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # FIGURE 7: State-Level Stunting & Wasting Variation (Selected Contrasting States)
    state_df = pd.DataFrame(state_records).sort_values("unweighted_stunting_pct", ascending=True)
    # Select 8 lowest and 8 highest stunting states for clean sample comparison
    sample_states = pd.concat([state_df.head(8), state_df.tail(8)])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_s = np.arange(len(sample_states))
    ax.barh(y_s - 0.18, sample_states["unweighted_stunting_pct"], 0.36, label="Stunting (Unweighted Cohort %)", color="#2980b9")
    ax.barh(y_s + 0.18, sample_states["unweighted_wasting_pct"], 0.36, label="Wasting (Unweighted Cohort %)", color="#e74c3c")
    ax.set_yticks(y_s)
    ax.set_yticklabels(sample_states["state_name"], fontsize=9)
    ax.set_xlabel("Unweighted Analytical-Cohort Prevalence (%)")
    ax.set_title("Figure 7: Unweighted Analytical-Cohort Prevalence Across Selected Indian States/UTs", fontsize=11)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig7_path = os.path.join(OUTPUT_DIR_FIGURES, "fig7_state_level_stunting_wasting.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 6. SAVE STRUCTURED EDA SUMMARY JSON
    # -------------------------------------------------------------------------
    eda_summary = {
        "dataset_total_records": total_records,
        "target_summary": target_summary,
        "bivariate_summary": bivariate_summary,
        "cramers_v_associations": cramers_matrix,
        "state_records": state_records,
        "figures_generated": [
            "fig1_target_prevalence_and_imbalance.png",
            "fig2_missingness_profile.png",
            "fig3_age_dynamics_by_outcome.png",
            "fig4_socioeconomic_maternal_gradients.png",
            "fig5_morbidity_associations_acute_vs_chronic.png",
            "fig6_statistical_association_cramers_v.png",
            "fig7_state_level_stunting_wasting.png"
        ]
    }
    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"[+] EDA summary successfully saved to {OUTPUT_SUMMARY_JSON}")
    print(f"[+] 7 figures successfully saved to {OUTPUT_DIR_FIGURES}/")
    print("[OK] Step 6 EDA completed successfully.")
    return eda_summary

if __name__ == "__main__":
    run_eda()
