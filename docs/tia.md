# Background

this package is designed targeting for TIA (time-integrated-activity) part of dosimetry calculation worklfow for radionuclides pharmaceutical injected into human body for therapy purpose. radionuclides pharmaceutical's distribution in human body can be imaged by PET or SPECT. And multi-timepoint activity maps are measured for TIA calculation and finally the dosimetry dosemap. In TIA (accumulated activity) calculation, there are various of fitting methods available , and the activity vs. time after injection variation among  organs and lesions are big. And the final TIA has been impacted by noise levels, fitting algorithms, and number of timepoints.

# Requirements
Search for an algorithm to handle:
1. noise level
2. support two or more timepoints
3. fitting methods depending on uptake curve, and fitting uncertainty estimation
4. physical decay is naturally embedded or can be an factor to constraint the curve fitting
5. modeling uptake and clearance, and estimate the peak time for each curve
6. matrix operations and inv operations for fitting curve
7. curves are classified based on the property of curve (different combination types of uptake + clearance)
8. noise floor consideration and reducing the noise
9. robustness to noise and types of curve and number of time points
10. Input and output: activity map (unit:Bq/ml) in nifti format or nibabel image, and time after injection;  TIA (unit: Bq*s) in nifti format or nibabel image, and share the same coordinates as activity. output TIA, goodness of fitting R^2, uncertainty maps
11. fitting methods are not limited to one, and user can make selection and also a recommendated fitting strategy is appliable as auto mode.
    
# Task
write a python package for this processing. Using nibabel to handle nifti files and this package accept nifti image files or nibabel images, and other information are provided by yaml config file or yaml config. this is for developing a python package called PyTIA, and please generate design document for me to review , once the review shows that the design is ready, then move to next step for develop, testing, validation and documentation.


# Final Design Document: Voxel-Wise Radionuclide Dosimetry Package

## 1. Executive Summary
This document specifies the architecture for a Python package designed to calculate Time Integrated Activity (TIA) maps from multi-timepoint PET/SPECT activity maps. The design prioritizes **robustness** against high noise levels, **adaptability** to varying number of timepoints ($N \ge 2$), and **physiological accuracy** by embedding physical decay constraints and using shape-priors for kinetic modeling.

## 2. System Architecture
The processing pipeline is divided into four sequential modules:

1.  **Data Ingestion & Masking:** Loading 4D NIfTI data and generating a body mask to exclude background air.
2.  **Spatial Regularization (Denoising):** Anatomically constrained Gaussian smoothing.
3.  **Kinetic Modeling Engine:** An adaptive decision tree that selects the optimal fitting strategy (Gamma Variate, Mono-exponential, or Hybrid) based on data availability and curve shape.
4.  **Integration & Map Generation:** Calculation of TIA (Bq$\cdot$s), effective decay rates, and uncertainty metrics.

---

## 3. Algorithm Design & Logic

### 3.1 Pre-processing: Robust Noise Handling (Req 2.1 & 3)
PET/SPECT images suffer from high voxel-wise noise (Poisson noise). Direct fitting on raw voxels leads to instability.

*   **Strategy:** **Masked Spatial Gaussian Smoothing**.
*   **Logic:**
    1.  Generate a binary `BodyMask` using Otsu thresholding on the Sum-Image.
    2.  Apply Gaussian smoothing ($\sigma \approx 1\text{-}2$ voxels) *only* within the mask.
    3.  **Benefit:** This reduces noise variance to stabilize curve fitting while preventing the "bleeding" of tumor activity into the zero-background air (Partial Volume Effect mitigation).

### 3.2 The Adaptive Kinetic Logic (Req 2.2 & 4)
To handle the variation between fast uptake/washout (Kidneys) and slow accumulation (Tumors), and to handle $N=2$ vs $N \ge 3$, the system uses a **Voxel-wise Decision Tree**.

For every voxel, the algorithm classifies the Time-Activity Curve (TAC) shape and routes it to one of three solvers:

#### **Model A: Gamma-Variate Fit (The "Optimized Peak" Model)**
*   **Trigger:** $N \ge 3$ **AND** Curve shape is "Hump" (starts low, rises, falls).
*   **Equation:** $A(t) = K \cdot t^\alpha \cdot e^{-\beta t}$
*   **Purpose:** Mathematically predicts the *true* peak time ($t_{peak} = \alpha/\beta$) and magnitude, which may occur between measured timepoints. Models fast uptake ($t^\alpha$) and clearance simultaneously.
*   **Constraint:** $\beta \ge \lambda_{phys}$ (Clearance cannot be slower than physical decay).
*   **Fallback:** If the fit fails to converge (due to noise), automatically downgrade to Model B.

#### **Model B: Constrained Mono-Exponential (The "Clearance" Model)**
*   **Trigger:**
    *   Curve shape is "Falling" (Peak at $t_1$, e.g., Kidneys).
    *   $N=2$ (Falling).
    *   Fallback from Model A.
*   **Logic:**
    1.  **Uptake Phase:** Assume Linear Uptake from $t=0$ to $t_{start}$.
    2.  **Clearance Phase:** Fit $A(t) = A_0 e^{-\lambda_{eff} t}$ to points $t \ge t_{peak}$.
*   **Physics Constraint:** $\lambda_{eff} = \max(\lambda_{fit}, \lambda_{phys})$.
*   **Handling $N=2$:** Uses analytical log-slope calculation instead of iterative fitting, clamped by $\lambda_{phys}$.

#### **Model C: Hybrid Method (Trapezoid + Tail)**
*   **Trigger:** User preference or complex multi-phasic curves where parametric models fail.
*   **Logic:**
    1.  **Observed:** Trapezoidal integration ($0 \to t_{last}$).
    2.  **Tail:** Extrapolate from $t_{last}$ using $\lambda_{phys}$ (conservative) or fitted tail slope.

### 3.3 Physical Constraints (Req 2.4)
The algorithm strictly enforces physical limits to prevent non-biological results (e.g., negative half-lives or accumulation exceeding physical decay limits).
*   **Rule:** $\lambda_{effective} \ge \lambda_{physical}$
*   **Implementation:**
    *   In Optimization: Use `bounds` in `scipy.optimize.curve_fit`.
    *   In Analytical N=2: Use `numpy.maximum(calculated_lambda, phys_lambda)`.

---

## 4. Detailed Data Flow

### Step 1: Input Analysis
*   Load NIfTI files.
*   Sort by timepoint.
*   Extract `Time` vector.
*   Define `Lambda_Phys = ln(2) / Half_Life`.

### Step 2: Voxel-wise Execution (Parallelized)
Iterate over all masked voxels (utilizing Multiprocessing Pool):

1.  **Shape Detection:**
    *   Identify index of max value ($idx_{max}$).
    *   Is $idx_{max} == 0$? $\rightarrow$ **Falling**.
    *   Is $idx_{max} == last$? $\rightarrow$ **Rising**.
    *   Else $\rightarrow$ **Hump**.

2.  **Algorithm Selection & Solving:**

    *   **IF (Rising):**
        *   Cannot determine biological clearance.
        *   Action: Trapezoid ($0 \to t_{last}$) + Physical Tail Integration.
        *   Uncertainty: High.

    *   **IF (Hump AND $N \ge 3$):**
        *   Attempt **Gamma Variate Fit**.
        *   If success: Integrate analytically $\int_0^\infty K t^\alpha e^{-\beta t} dt$.
        *   If fail: Treat as "Falling" starting from $idx_{max}$.

    *   **IF (Falling OR Fallback):**
        *   **Uptake Area:** $0.5 \times A(t_{start}) \times t_{start}$ (Triangle approximation).
        *   **Clearance Area:** Fit exponential to $t[idx_{max}:]$.