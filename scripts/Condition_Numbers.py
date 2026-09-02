import importlib # for changes and debugging
import safari
import safari.Frame_Builder as fb
import safari.SSM_Builder as ssm
# Reload deepest dependency first, then up the chain
importlib.reload(fb)
importlib.reload(ssm)
importlib.reload(safari)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

L = 2**14

# list of sizes of frames to generate
N_list = [8,16,26,32,48,64]
cols = ['LegS_cond', 'LegS_neff', 'LegT_cond', 'LegT_neff', 'FouS_cond', 'FouS_neff',
        'FouT_cond', 'FouT_neff', 'ChebyS_cond', 'ChebyS_neff', 'ChebyT_cond', 'ChebyT_neff',
        'LagS_cond', 'LagS_neff', 'LagT_cond', 'LagT_neff', 'BernS_cond', 'BernS_neff', 
        'BernT_cond', 'BernT_neff', 'GabS_cond', 'GabS_neff', 'GabT_cond', 'GabT_neff',
        'DaubS_cond', 'DaubS_neff', 'DaubT_cond', 'DaubT_neff']

Cond_num = pd.DataFrame(np.nan, index=range(len(N_list)), columns=['N'] + cols)

for i, N in enumerate(N_list):
    LegS = safari.SSM(N=N, L=L, fname='legendre', meas='scaled')
    LegT = safari.SSM(N=N, L=L, fname='legendre', meas='translated')
    FouS = safari.SSM(N=N, L=L, fname='fourier', meas='scaled')
    FouT = safari.SSM(N=N, L=L, fname='fourier', meas='translated')
    ChebyS = safari.SSM(N=N, L=L, fname='chebyshev', meas='scaled')
    ChebyT = safari.SSM(N=N, L=L, fname='chebyshev', meas='translated')
    LagS = safari.SSM(N=N, L=L, fname='laguerre', meas='scaled')
    LagT = safari.SSM(N=N, L=L, fname='laguerre', meas='translated')
    BernS = safari.SSM(N=N, L=L, fname='bernstein', meas='scaled')
    BernT = safari.SSM(N=N, L=L, fname='bernstein', meas='translated')
    GabS = safari.SSM(N=N, L=L, fname='gabor', meas='scaled', m=4, redundancy=0.1)
    GabT = safari.SSM(N=N, L=L, fname='gabor', meas='translated', m=4, redundancy=0.1)
    DaubS = safari.SSM(N=N, L=L, fname='daubechies', meas='scaled', dborder='db6')
    DaubT = safari.SSM(N=N, L=L, fname='daubechies', meas='translated', dborder='db6')

    Cond_num.loc[i] = [N, LegS.cond, LegS.erank, LegT.cond, LegT.erank, FouS.cond, FouS.erank,
        FouT.cond, FouT.erank, ChebyS.cond, ChebyS.erank, ChebyT.cond, ChebyT.erank,
        LagS.cond, LagS.erank, LagT.cond, LagT.erank, BernS.cond, BernS.erank, 
        BernT.cond, BernT.erank, GabS.cond, GabS.erank, GabT.cond, GabT.erank, 
        DaubS.cond, DaubS.erank, DaubT.cond, DaubT.erank]

x_axis = 'N'
cond_columns = [col for col in Cond_num.columns if '_cond' in col]

# 3. Create the Plot
plt.figure(figsize=(10, 6))

color_map = {} 
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_idx = 0

# 3. Generate the Plot
for col in cond_columns:
    # A. Extract everything before '_cond' (e.g., 'LegS_cond' -> 'LegS')
    prefix = col.split('_cond')[0]

    # B. Robustly identify variant (S or T) by checking the LAST letter
    if prefix.endswith('S'):
        base_name = prefix[:-1]  # Slice off the last letter ('LegS' -> 'Leg')
        line_style = '-'         # Solid
        variant_label = '(S)'
    elif prefix.endswith('T'):
        base_name = prefix[:-1]  # Slice off the last letter ('LegT' -> 'Leg')
        line_style = '--'        # Dashed
        variant_label = '(T)'
    else:
        base_name = prefix
        line_style = ':'         
        variant_label = '(Other)'

    # C. Assign or Retrieve Color for the Base Name
    if base_name not in color_map:
        color_map[base_name] = default_colors[color_idx % len(default_colors)]
        color_idx += 1
    
    component_color = color_map[base_name]

    # D. Plot Line
    plt.plot(Cond_num[x_axis], Cond_num[col], 
             linestyle=line_style, 
             color=component_color, 
             marker='o', 
             linewidth=1.5, 
             label=f"{base_name} {variant_label}")

# 4. Finalize Styling
plt.xlabel('N', fontsize=12)
plt.ylabel('Condition Number', fontsize=12)
plt.yscale('log')
plt.title('Condition Numbers of SSM A Matrices (Color = Frame, Window = S -, T --)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Components", frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()