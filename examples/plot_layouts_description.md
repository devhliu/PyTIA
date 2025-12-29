# Lu-177 PSMA Phantom - Plot Layouts and Interpretation Guide

## Generated Plots Overview

The plotting script generates 8 detailed PNG files showing the TIA fitting analysis:

### 1. lu177_psma_result_maps.png
**Layout**: 2×3 subplot grid
- **Top Left**: TIA map showing integrated activity (hot colormap)
- **Top Middle**: R² map showing goodness of fit (viridis colormap, 0-1)
- **Top Right**: Status map showing fit success/failure (categorical colors)
- **Bottom Left**: Anatomical overlay showing organ positions
- **Bottom Middle**: Histogram of all valid TIA values
- **Bottom Right**: Summary statistics table for all organs

### 2. Organ-Specific Fitting Analysis (liver_fitting_analysis.png, kidneys_fitting_analysis.png, marrow_fitting_analysis.png, tumors_fitting_analysis.png)
**Layout**: 2×2 subplot grid for each organ
- **Top Left**: Time-Activity Curves with sample voxels
  - Black dashed line: True theoretical curve
  - Colored lines: Individual voxel measurements (colored by R²)
  - Shows 6 time points at 4, 24, 48, 96, 112, 176 hours
- **Top Right**: R² distribution histogram
  - Shows distribution of fit quality across all voxels
  - Red line indicates mean R² value
- **Bottom Left**: TIA distribution comparison
  - Green histogram: Measured TIA values
  - Red dashed line: Theoretical TIA from true curve
  - Blue dashed line: Mean measured TIA
- **Bottom Right**: TIA vs R² scatter plot
  - Reveals correlation between fit quality and TIA accuracy
  - Shows Pearson correlation coefficient

### 3. Individual Tumor Analysis (tumor_1_detailed_analysis.png, tumor_2_detailed_analysis.png, tumor_3_detailed_analysis.png)
Same 2×2 layout as organ plots but focused on individual lesions, allowing detailed examination of:
- Lesion-specific kinetics
- Heterogeneity within each tumor
- Individual tumor response patterns

## Key Visual Indicators

### Color Maps
- **TIA Map**: Red = high integrated activity, Black = low/none
- **R² Map**: Yellow = excellent fit (1.0), Purple = poor fit (<0.5)
- **Status Map**:
  - Black (0): Outside mask
  - Green (1): Successful fit
  - Yellow (2): <2 valid points
  - Red (3): Fit failed
  - Orange (4): All points below noise floor
  - Purple (5): Non-physical parameters

### Anatomy Colors
- Blue: Liver
- Red: Kidneys
- Green: Tumors
- Yellow: Red marrow

## Interpretation Guidelines

### Good Fitting Indicators
- R² > 0.90 in high-activity regions
- Measured TIA close to theoretical value
- Low scatter in TIA distribution (<10% CV)

### Potential Issues
- Low R² in high-activity regions → model mismatch
- High status failure rates → adjust parameters
- Large TIA variance → check segmentation or noise levels