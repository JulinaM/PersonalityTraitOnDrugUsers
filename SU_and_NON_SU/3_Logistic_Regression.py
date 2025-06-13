import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_log_odds(results_df, title, filename):
    """
    Generates and saves a plot of log odds ratios with 95% confidence intervals.

    Args:
        results_df (pd.DataFrame): DataFrame with columns 'param', 'coef', 'conf_lower', 'conf_upper'.
        title (str): The title for the plot.
        filename (str): The path to save the plot image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the error bar plot
    ax.errorbar(
        x=results_df['coef'],
        y=results_df['param'],
        xerr=[results_df['coef'] - results_df['conf_lower'], results_df['conf_upper'] - results_df['coef']],
        fmt='o',
        color='darkblue',
        ecolor='skyblue',
        elinewidth=3,
        capsize=5,
        markersize=8
    )

    # Add a vertical line at 0 for reference
    ax.axvline(x=0, linestyle='--', color='grey', linewidth=1)

    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Log Odds Ratio (Coefficient)', fontsize=12)
    ax.set_ylabel('Personality Trait', fontsize=12)
    
    # Invert y-axis to have a more intuitive top-to-bottom reading
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to: {filename}")


def run_personality_substance_use_analysis():
    """
    This script performs a logistic regression analysis to determine if
    personality traits are associated with substance use (SU).
    """
    print("--- Starting Logistic Regression Analysis ---")

    # --- Configuration ---
    BASE_PATH = "/data2/julina/scripts/tweets/2020/03/"
    ANALYSIS_DIR = os.path.join(BASE_PATH, "SU_and_NON_SU_analysis/")
    INPUT_FILE = os.path.join(ANALYSIS_DIR, "all_users_classified_with_personality.csv")
    
    PERSONALITY_TRAITS = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
    TARGET_VARIABLE = 'DrugAbuse'
    
    # Dictionary to map short trait names to full names for plotting
    trait_name_map = {
        'cOPN': 'Openness',
        'cCON': 'Conscientiousness',
        'cEXT': 'Extraversion',
        'cAGR': 'Agreeableness',
        'cNEU': 'Neuroticism'
    }

    # --- Step 1: Load Data ---
    print(f"\nLoading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"FATAL ERROR: Input file not found at {INPUT_FILE}.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=PERSONALITY_TRAITS + [TARGET_VARIABLE], inplace=True)
    print(f"Data shape after dropping NA: {df.shape}")

    # --- Step 2: Balance the Dataset ---
    df_su = df[df[TARGET_VARIABLE] == 1]
    df_non_su = df[df[TARGET_VARIABLE] == 0]

    print(f"\nOriginal class distribution:")
    print(f"  SU Users (1): {len(df_su)}")
    print(f"  NON-SU Users (0): {len(df_non_su)}")

    if len(df_su) > len(df_non_su):
        majority_df, minority_df = df_su, df_non_su
    else:
        majority_df, minority_df = df_non_su, df_su

    majority_downsampled = resample(
        majority_df, replace=False, n_samples=len(minority_df), random_state=42
    )
    df_balanced = pd.concat([minority_df, majority_downsampled])
    
    print("\nBalanced class distribution for analysis:")
    print(df_balanced[TARGET_VARIABLE].value_counts())
    
    # --- Step 3: Define Variables ---
    y = df_balanced[TARGET_VARIABLE]
    X_multi = sm.add_constant(df_balanced[PERSONALITY_TRAITS])

    # --- Step 4: Multiple Logistic Regression ---
    print("\n\n" + "="*60)
    print("  Multiple Logistic Regression: All Traits vs. Substance Use")
    print("="*60)
    results_for_plotting_multi = pd.DataFrame()
    try:
        logit_model_all = sm.Logit(y, X_multi).fit()
        print(logit_model_all.summary())
        
        # Store results for plotting, dropping the 'const'
        params = logit_model_all.params.drop('const')
        conf = logit_model_all.conf_int().drop('const')
        
        results_for_plotting_multi = pd.DataFrame({
            "param": params.index,
            "coef": params.values,
            "conf_lower": conf[0].values,
            "conf_upper": conf[1].values
        })
        # **FIX:** Explicitly map the short names to full names for the plot
        results_for_plotting_multi['param'] = results_for_plotting_multi['param'].map(trait_name_map)
    except Exception as e:
        print(f"Could not run multiple logistic regression. Error: {e}")

    # --- Step 5: Individual Logistic Regressions ---
    print("\n\n" + "="*60)
    print("  Individual Logistic Regressions: Each Trait vs. Substance Use")
    print("="*60)
    individual_results_list = []
    for trait in PERSONALITY_TRAITS:
        try:
            X_single = sm.add_constant(df_balanced[trait])
            logit_model_single = sm.Logit(y, X_single).fit(disp=0)
            
            coef = logit_model_single.params.values[1]
            conf_lower, conf_upper = logit_model_single.conf_int().iloc[1]
            
            individual_results_list.append({
                "param": trait, # Keep short name for now
                "coef": coef, 
                "conf_lower": conf_lower, 
                "conf_upper": conf_upper
            })
        except Exception as e:
            print(f"Could not run regression for {trait}. Error: {e}")
            
    results_for_plotting_individual = pd.DataFrame(individual_results_list)
    # **FIX:** Explicitly map the short names to full names for the plot
    if not results_for_plotting_individual.empty:
        results_for_plotting_individual['param'] = results_for_plotting_individual['param'].map(trait_name_map)

    # --- Step 6: Generate Plots ---
    print("\n\n" + "="*60)
    print("  Generating and Saving Log Odds Ratio Plots")
    print("="*60)

    # Plot for Multiple Regression Model
    if not results_for_plotting_multi.empty:
        print("\nData being sent to the MULTIPLE regression plot:")
        print(results_for_plotting_multi)
        plot_log_odds(
            results_df=results_for_plotting_multi,
            title='Log Odds Ratios of Personality Traits on Substance Use\n(Multiple Regression Model)',
            filename=os.path.join(ANALYSIS_DIR, 'multi_regression_log_odds.png')
        )
    
    # Plot for Individual Regression Models
    if not results_for_plotting_individual.empty:
        print("\nData being sent to the INDIVIDUAL regression plot:")
        print(results_for_plotting_individual)
        plot_log_odds(
            results_df=results_for_plotting_individual,
            title='Log Odds Ratios of Personality Traits on Substance Use\n(Individual Regression Models)',
            filename=os.path.join(ANALYSIS_DIR, 'individual_regression_log_odds.png')
        )

    print("\n\n--- Analysis Complete ---")

if __name__ == '__main__':
    run_personality_substance_use_analysis()
