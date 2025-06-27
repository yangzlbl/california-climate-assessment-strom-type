import pandas as pd

# df_max = pd.read_csv('/.../region_max_precipitation_all.csv')
df_ave = pd.read_csv('/.../LOCA2_hybrid_region_ave_precipitation_all.csv')

# Load full dataset again for a clean start
df = df_ave

# Define the region of interest
region_focus = "North Coast"   # Change to other regions

# Categorize scenario groups
def categorize_scenario_fixed(row):
    if row['Scenario'] == 'historical' and 1980 <= row['Year'] <= 2010:
        return 'Historical (1980–2010)'
    elif row['Scenario'] == 'ssp585-mid' and 2040 <= row['Year'] <= 2070:
        return 'SSP585 Mid (2040–2070)'
    elif row['Scenario'] == 'ssp585-end' and 2070 <= row['Year'] <= 2100:
        return 'SSP585 End (2070–2100)'
    return None

# Parse dates and compute water year and day of season
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Scenario_Group'] = df.apply(categorize_scenario_fixed, axis=1)
df = df[df['Scenario_Group'].notnull() & df['Landfall'].isin([0, 1, 2])]

# Keep relevant columns only
df = df[['Date', 'Year', 'Month', 'Scenario_Group', 'Landfall', 'Model', region_focus]].rename(
    columns={region_focus: 'Precipitation'}
)

# Assign water year
df['Water_Year'] = df.apply(
    lambda row: row['Year'] + 1 if row['Month'] in [10, 11, 12] else row['Year'],
    axis=1
)

# Calculate day in water year (Oct 1 = Day 1)
df['Day_of_Season'] = (
    df['Date'] - pd.to_datetime(df['Water_Year'] - 1, format="%Y") - pd.Timedelta(days=273)
).dt.days + 1
df = df[(df['Day_of_Season'] >= 1) & (df['Day_of_Season'] <= 365)]

# Label storm types
df['Landfall_Type'] = df['Landfall'].map({0: 'Non-AR', 1: 'AR-only', 2: 'AR-ETC'})

# Step 1: Compute daily mean per model, storm type, scenario group, and day
daily_means = df.groupby(
    ['Scenario_Group', 'Water_Year', 'Day_of_Season', 'Model', 'Landfall_Type']
)['Precipitation'].sum().unstack(fill_value=0).reset_index()

# Step 2: Average across models
daily_ensemble_mean = daily_means.groupby(
    ['Scenario_Group', 'Day_of_Season']
).mean().reset_index()

# Step 3: Compute cumulative sum
for col in ['Non-AR', 'AR-only', 'AR-ETC']:
    daily_ensemble_mean[col] = daily_ensemble_mean.groupby('Scenario_Group')[col].cumsum()

# Total cumulative
daily_ensemble_mean['Total'] = daily_ensemble_mean[['Non-AR', 'AR-only', 'AR-ETC']].sum(axis=1)

# Plot
import matplotlib.pyplot as plt

scenarios = ['Historical (1980–2010)', 'SSP585 Mid (2040–2070)', 'SSP585 End (2070–2100)']
fig, axes = plt.subplots(1, len(scenarios), figsize=(18, 5), sharey=True)

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    ax.grid(linestyle=':',color='gray',alpha=0.7,zorder=0)
    sub = daily_ensemble_mean[daily_ensemble_mean['Scenario_Group'] == scenario].sort_values('Day_of_Season')

    x = sub['Day_of_Season'].values
    y_non_ar = sub['Non-AR'].values
    y_ar_only = sub['AR-only'].values
    y_ar_etc = sub['AR-ETC'].values
    y_total = sub['Total'].values

    ax.stackplot(x, y_non_ar, y_ar_only, y_ar_etc,
                 labels=['Non-AR', 'AR-only', 'AR-ETC'], alpha=0.6)
    ax.plot(x, y_total, color='black', linewidth=2, label='Total Precipitation')

    ax.set_xlim(1, 365)
    ax.set_title(f"{region_focus} - {scenario}")
    ax.set_xlabel("Day in Water Year (Oct 1 = Day 1)")
    if i == 0:
        ax.set_ylabel("Cumulative Precipitation (mm)")
    ax.legend(loc='upper left')
    

fig.suptitle("North Coast: Multi-Model Mean Cumulative Water Year Precipitation by Storm Type", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


