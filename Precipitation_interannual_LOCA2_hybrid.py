import matplotlib.pyplot as plt
# Re-load the uploaded dataset
df_ave = pd.read_csv('/.../LOCA2_hybrid_region_ave_precipitation_all.csv')
df = df_ave

# Repeat the previous analysis steps
# Define region columns
region_columns = [
    'Inland South', 'Los Angeles', 'San Diego', 'San Francisco Bay Area',
    'San Joaquin Valley', 'North Coast', 'Sacramento Valley', 'Desert',
    'Central Coast'
]

# Assign scenario group
def categorize_scenario_fixed(row):
    if row['Scenario'] == 'historical' and 1980 <= row['Year'] <= 2010:
        return 'Historical (1980–2010)'
    elif row['Scenario'] == 'ssp585-mid' and 2040 <= row['Year'] <= 2070:
        return 'SSP585 Mid (2040–2070)'
    elif row['Scenario'] == 'ssp585-end' and 2070 <= row['Year'] <= 2100:
        return 'SSP585 End (2070–2100)'
    return None

# Add time-related columns
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Scenario_Group'] = df.apply(categorize_scenario_fixed, axis=1)

# Filter for ONDJFM and valid values
# winter_months = [10, 11, 12, 1, 2, 3]
# df = df[df['Month'].isin(winter_months) & df['Scenario_Group'].notnull() & df['Landfall'].isin([0, 1, 2])]
df['Landfall_Type'] = df['Landfall'].map({0: 'Non-AR', 1: 'AR-only', 2: 'AR-ETC'})
df['Water_Year'] = df.apply(
    lambda row: row['Year'] + 1 if row['Month'] in [10, 11, 12] else row['Year'],
    axis=1
)
# Melt long for extreme identification
melted = df.melt(
    id_vars=['Water_Year', 'Scenario_Group', 'Landfall_Type', 'Model'],
    value_vars=region_columns,
    var_name='Region',
    value_name='Precipitation'
)

# Average daily precipitation over the ONDJFM months, grouped by year, region, type, model
annual_mean = melted.groupby(['Water_Year', 'Region', 'Landfall_Type', 'Model'])['Precipitation'].sum().reset_index()

# Then average across models to get ensemble mean
annual_ensemble_mean = annual_mean.groupby(['Water_Year', 'Region', 'Landfall_Type'])['Precipitation'].mean().reset_index()

# Filter to a single region for stacked bar plotting
region_focus = "San Francisco Bay Area"
annual_region = annual_ensemble_mean[annual_ensemble_mean['Region'] == region_focus]

# Pivot for plotting
pivot_annual = annual_region.pivot(index='Water_Year', columns='Landfall_Type', values='Precipitation').fillna(0)

# Group by water year and find the dominant scenario group
year_scenarios = df.groupby('Water_Year')['Scenario_Group'].agg(lambda x: x.mode()[0]).reset_index()

valid_years = year_scenarios.groupby('Scenario_Group')['Water_Year'].agg(['min', 'max']).reset_index()
mask = pd.Series(False, index=year_scenarios.index, dtype=bool)
for _, row in valid_years.iterrows():
    scenario = row['Scenario_Group']
    year_min = row['min']
    year_max = row['max']
    mask |= ((year_scenarios['Scenario_Group'] == scenario) &
             (year_scenarios['Water_Year'] > year_min) &
             (year_scenarios['Water_Year'] < year_max))

cleaned_years = year_scenarios[mask]['Water_Year']
pivot_annual_cleaned = pivot_annual.loc[cleaned_years]

# Merge scenario group into pivot_annual for plotting
pivot_annual_with_scenarios = pivot_annual_cleaned.merge(year_scenarios, left_index=True, right_on='Water_Year')
pivot_annual_with_scenarios.sort_values('Water_Year', inplace=True)

# Start plotting
fig, ax = plt.subplots(figsize=(18, 6))

# Stacked bar chart
pivot_annual_with_scenarios.set_index('Water_Year')[['Non-AR', 'AR-only', 'AR-ETC']].plot(
    kind='bar',
    stacked=True,
    ax=ax,
    width=0.9
)

# Scenario group color mapping
scenario_colors = {
    'Historical (1980–2010)': '#f0f0f0',
    'SSP585 Mid (2040–2070)': '#ffe6e6',
    'SSP585 End (2070–2100)': '#ffcccc'
}

# Plot shaded backgrounds for each scenario block
years = pivot_annual_with_scenarios['Water_Year'].values
scenario_blocks = pivot_annual_with_scenarios['Scenario_Group'].values

prev_idx = 0
for i in range(1, len(scenario_blocks)):
    if scenario_blocks[i] != scenario_blocks[prev_idx]:
        ax.axvspan(prev_idx - 0.5, i - 0.5, color=scenario_colors[scenario_blocks[prev_idx]], alpha=0.3)
        prev_idx = i
# Final band
ax.axvspan(prev_idx - 0.5, len(years) - 0.5, color=scenario_colors[scenario_blocks[prev_idx]], alpha=0.3)

# Add vertical lines at transitions
for i in range(1, len(scenario_blocks)):
    if scenario_blocks[i] != scenario_blocks[i - 1]:
        ax.axvline(i - 0.5, color='black', linestyle='--', linewidth=3)

# Final touches
ax.set_title(f"Annual Precipitation in {region_focus} by Storm Type and Scenario", fontsize=14)
ax.set_ylabel("Ensemble Mean Total Precipitation (mm)")
ax.set_xlabel("Water Year")
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=90)
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.grid(linestyle=":",color="gray",alpha=0.7,zorder=0)
plt.show()