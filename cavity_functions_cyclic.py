"""
@author: felixgrandadam
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pyvista as pv
import pandas as pd
import os

from matplotlib.colors import LogNorm, Normalize
import matplotlib.patches as patches

from scipy.ndimage import label, binary_erosion, binary_dilation, convolve, zoom, gaussian_filter,distance_transform_edt, binary_opening
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

import vtk
vtk.vtkObject.GlobalWarningDisplayOff()


#by default : 
threshold = 0.15 #m 




##############################
# CSV pre-processing : 
##############################



# compute the cavitation state theta from the stress field and basal velocity in the dataset :
def compute_cavitation(data_c,K_wi_c): 

    m=3
    rho_i = 920  # m3/kg
    g = 9.81  # m/s2
    H=230 # m (from Gagliardini, it's the height of the ice slab )
    
    tau_b_chosen = np.abs(data_c['stress vector_0']*1e6).mean() # from Mpa to Pa
    U_b_chosen = (data_c['velocity_0']/365.25/86400).mean()     # from m/a to m/s
    
    tau_b_0_pow_m_list = []
    U_b_0_list = []
    
    zero_overburden_ratio_K_wi_list = np.arange(0,1,1) # overburden pressure ratio settings for which there is no cavitation, 
                                                     # here we consider this True for ovb_ratio = 0 only
    for K_wi in zero_overburden_ratio_K_wi_list :
        K_wi_tmp = str(K_wi)
        if len(K_wi_tmp) ==1 :
            K_wi_tmp = '0'+ K_wi_tmp
        tmp_filename = 'sinus3dnS_1_14_'+K_wi_tmp+'_0065__t0002'
        tmp_path_name='../../data/sinus3dnS_1_14_0065/'+tmp_filename+'.csv'
        tmp_data=pd.read_csv(tmp_path_name)
        tau_b_0_pow_m_list.append(pow(np.abs(tmp_data['stress vector_0']*1e6).mean(),m))
        U_b_0_list.append((tmp_data['velocity_0']/365.25/86400).mean())

    # when there is no cavitation we can infer A_s :
    A_s = (np.array(U_b_0_list)/np.array(tau_b_0_pow_m_list)).mean() # we consider cavitation = 0. 
    
    print(f'The found A_s is : {A_s:.4e} m s⁻¹ Pa^(-m)')
    
    if int(K_wi_c) in zero_overburden_ratio_K_wi_list :
        cavitation_chosen = 0
    else :
        cavitation_chosen = 1 - (tau_b_chosen**m * A_s/U_b_chosen)


    N_norm_chosen = (1-int(K_wi_c)/100)*H*rho_i*g # N = Pi - Pw   <=>   N = Pi (1 - K_wi_c ) 

    # get all the K_pi/pw
    known_overburden_ratio_K_wi_list = np.arange(0,81,1)  # overburden pressure ratio settings for which there is a water discharge/path.
    
    cavitation_list = []
    N_norm_list = []
    tau_b_list  = []
    U_b_list    = []
    h_list      = []
    
    for K_wi in known_overburden_ratio_K_wi_list :
        K_wi_tmp = str(K_wi)
        if len(K_wi_tmp) ==1 :
            K_wi_tmp  = '0'+ K_wi_tmp
        tmp_filename  = 'sinus3dnS_1_14_'+K_wi_tmp+'_0065__t0002'
        tmp_path_name = '../../data/sinus3dnS_1_14_0065/'+tmp_filename+'.csv'
        tmp_data = pd.read_csv(tmp_path_name)
        tau_b = np.abs(tmp_data['stress vector_0']*1e6).mean()
        U_b   = (tmp_data['velocity_0']/365.25/86400).mean()
        U_b_list.append(U_b)
        tau_b_list.append(tau_b)
        N_norm = (1-int(K_wi_tmp)/100)*rho_i*g*H
        N_norm_list.append(N_norm)
    
        h_list.append(tmp_data.hcavity.mean())
        
        cavitation = 1 - (tau_b**m * A_s/U_b) # as defined in (1) Gilbert & al 2022
        cavitation_list.append(cavitation)


    plt.plot(known_overburden_ratio_K_wi_list,cavitation_list,'o-',label='from (eq 7)')
    plt.axvline(int(K_wi_c),linestyle='--', label=f'overburden ratio : {K_wi_c}')
    plt.axhline(cavitation_list[int(K_wi_c)],linestyle='--', label=f'cavitation : {cavitation_list[int(K_wi_c)]:.2f}')

    plt.xlabel(r'$K_{Pw/Pi}$',fontsize=16)
    plt.ylabel(r'taux de cavitation $\theta$',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.0,1.1)
    plt.legend()

    dic_values = {'h': h_list, 'N_norm': N_norm_list, 'tau_b': tau_b_list, 'U_b': U_b_list } # store all the spatial mean variables for all the K_wi

    return cavitation_chosen, cavitation_list, dic_values, tau_b_chosen, A_s




##############################
# CSV filtering : 
##############################




def correct_periodicity(df):
    vars_to_correct = ['zb', 'bedrock', 'hcavity']

    x_max = df['Points_0'].max()
    y_max = df['Points_1'].max()
    x_min = df['Points_0'].min()
    y_min = df['Points_1'].min()

    # correct X boundary :

    min_boundary_x = df[df['Points_0'] == x_min].sort_values('Points_1')[vars_to_correct].values
    max_indices_x = df[df['Points_0'] == x_max].sort_values('Points_1').index

    df.loc[max_indices_x, vars_to_correct] = min_boundary_x

    # correct Y boundary :

    min_boundary_y = df[df['Points_1'] == y_min].sort_values('Points_0')[vars_to_correct].values
    max_indices_y = df[df['Points_1'] == y_max].sort_values('Points_0').index

    df.loc[max_indices_y, vars_to_correct] = min_boundary_y


    return df



    

def dataset_3x3_tiling(ds):
    # Seamless tiling of the dataset : meaning we end up with :
    # a domain concatenated like so ABCD ABCD ABCD A

    # Calculate the coordinate step sizes (dx, dy)
    x_step = ds.Points_0.diff(dim="Points_0")[0].item()
    y_step = ds.Points_1.diff(dim="Points_1")[0].item()


    # Calculate the periodic lenght
    x_period = ds.Points_0.max().item() - ds.Points_0.min().item()
    y_period = ds.Points_1.max().item() - ds.Points_1.min().item() 

    # Create the 3x3 Grid via Concatenation after droping the last duplicated column/row
    ds_unique = ds.isel(Points_0=slice(0, -1),Points_1=slice(0, -1) )

    # Create the right-hand tile (shift along X)
    ds_right = ds_unique.copy()
    ds_right['Points_0'] = ds_right['Points_0'] + x_period
    
    ds_right_right = ds_right.copy()
    ds_right_right['Points_0'] = ds_right_right['Points_0'] + x_period

    # Concatenate original and right tile along the X dimension
    # This gives us the bottom half of our 3x3 grid (a 1x3 row)
    row1 = xr.concat([ds_unique, ds_right, ds_right_right], dim='Points_0')

    # Create the top half of the grid from the 3x1 (shift along Y)
    row2 = row1.copy()
    row2['Points_1'] = row2['Points_1'] + y_period

    row3 = row2.copy()
    row3['Points_1'] = row3['Points_1'] + y_period

    # Concatenate the bottom row and top row along the Y dimension
    ds_3x3_unique = xr.concat([row1, row2, row3], dim='Points_1')

    #copy first column :
    closing_col = ds_3x3_unique.isel(Points_0=0).copy()
    # extend the domain 1 step along Points_0 : 
    closing_col["Points_0"] = ds_3x3_unique.Points_0.max().item() + x_step
    #paste the first column to the end :
    ds_3x3_unique = xr.concat([ds_3x3_unique, closing_col], dim="Points_0")

    # do the same for rows :
    closing_row = ds_3x3_unique.isel(Points_1=0).copy()
    closing_row["Points_1"] = ds_3x3_unique.Points_1.max().item() + y_step
    
    final_ds = xr.concat([ds_3x3_unique, closing_row], dim="Points_1")

    return final_ds


def enforce_minimum_connection_width(ds):
    # 1. Extract the full binary mask of the cavity (both 0s and 1s)
    raw_mask = (ds['cavity_status'] >= 0).values
    
    # 2. Perform the opening to destroy narrow 1-pixel bridges
    kernel_2x2 = np.ones((2, 2), dtype=bool)
    clean_mask = binary_opening(raw_mask, structure=kernel_2x2)
    
    # 3. Recalculate inner (1) and boundary (0) so the cut edges heal correctly
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    new_inner = binary_erosion(clean_mask, structure=cross, border_value=1)
    
    # 4. Construct the new array using standard NumPy
    new_status = np.where(clean_mask, 0, -1)          # Lay down the boundaries
    new_status = np.where(new_inner, 1, new_status)   # Fill in the cores
    
    # 5. Safely plug it back into the dataset by explicitly passing the original dimensions
    ds['cavity_status'] = (ds['cavity_status'].dims, new_status)
    
    return ds


def force_periodicity(ds):
    # This function controls that the edges of our domain are periodic
    # has this may have been affected by optimize_bc()

    # fetch the 4 edges values array :
    first_row = ds["cavity_status"].isel(Points_1=0).values
    last_row  = ds["cavity_status"].isel(Points_1=-1).values
    first_column = ds["cavity_status"].isel(Points_0=0).values
    last_column  = ds["cavity_status"].isel(Points_0=-1).values

    #early check for periodicity :
    if np.array_equal(first_row,last_row) and np.array_equal(first_column,last_column) :
        #print('The periodicity has been conserved trough the rolling')
        return ds
    else : 
         print('The periodicity has NOT been conserved trough the rolling')
    # create masks where True if the values are common, False otherwise
    #row_mask = first_row == last_row
    #col_mask = first_column == last_column
    row_mask = np.isclose(first_row, last_row, equal_nan =True)
    col_mask = np.isclose(first_column, last_column, equal_nan =True)

    nb_of_items_drop = (~row_mask).sum() + (~col_mask).sum()
    print(f'number of cells dropped to re-establish periodicity : {nb_of_items_drop}')
    
    # update the DS : keep the value if true, switch to -1 otherwise :
    ds["cavity_status"].values[-1,:] = np.where(row_mask, last_row , -1)
    ds["cavity_status"].values[ 0,:] = np.where(row_mask, first_row, -1)
    ds["cavity_status"].values[:,-1] = np.where(col_mask, last_column , -1)
    ds["cavity_status"].values[:, 0] = np.where(col_mask, first_column, -1)

    ds["bedrock"].values[-1,:] = ds["bedrock"].values[0,:]
    ds["bedrock"].values[:,-1] = ds["bedrock"].values[:,0]
    
    ds["zb"].values[-1,:] = ds["zb"].values[0,:]
    ds["zb"].values[:,-1] = ds["zb"].values[:,0]
    
    return ds

    return ds




def periodic_value_roll(ds, roll_amount, dim_name):
    """
    Safely rolls a periodic xarray Dataset along a dimension, keeping the 
    spatial coordinates sorted and preserving duplicate endpoints.
    """
    if roll_amount == 0:
        return ds
    
    # 1. Back up the clean, monotonically increasing original coordinates
    orig_coords = ds[dim_name].values.copy()
    
    # 2. Slice off the duplicate endpoint (the last row/column)
    core_ds = ds.isel({dim_name: slice(0, -1)})
    
    # 3. Roll the unique interior data (ignore coord mapping for now)
    rolled_core = core_ds.roll({dim_name: roll_amount}, roll_coords=False)
    
    # 4. Re-create the closing duplicate element from the new first element
    new_last = rolled_core.isel({dim_name: 0})
    
    # 5. Stitch the core and the matching boundary back together
    import xarray as xr
    final_ds = xr.concat([rolled_core, new_last], dim=dim_name)
    
    # 6. Re-assign the pristine, sorted spatial coordinates
    final_ds[dim_name] = orig_coords
    
    return final_ds
    



def optimize_bc(ds):
    # This function take advantage of the periodic nature of our domain
    # by shifting it around to avoid physical no-sens

    # re-separate into groups the different clusters that we know we should keep :
    cavity_mask = ds.cavity_status >= 0
    mask_array = cavity_mask.values
    cross = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]])
    labeled, _   = label(cavity_mask, structure=cross)
    #plt.imshow(labeled)
    #plt.show()
    min_labels_count = 9999
    pts_0_to_roll = 0
    num_columns = labeled.shape[1]

    for x in range(num_columns):
        # cut the duplicated end column
        core_mask = mask_array[:, :-1]

        rolled_core = np.roll(core_mask, -x, axis=1)
        
        # Stitch the new first column onto the end to restore periodicity
        full_rolled_mask = np.concatenate((rolled_core, rolled_core[:, 0:1]), axis=1)
        
        labeled_rolled, _ = label(full_rolled_mask, structure = cross)

        boundary_slice = np.concatenate((labeled_rolled[:,0], labeled_rolled[:,-1] ))

        unique_labels_count = len(np.unique(boundary_slice))
        if unique_labels_count < min_labels_count :
            min_labels_count = unique_labels_count
            pts_0_to_roll = x
        if unique_labels_count <=2 :
            pts_0_to_roll = x
            break
    #plt.imshow(full_rolled_mask)
    #plt.show()
    ds = periodic_value_roll(ds, -pts_0_to_roll, 'Points_0')
    status_array = ds.cavity_status.values
    num_rows = status_array.shape[0]

    min_border_count = 9999
    max_outside_count = -1
    pts_1_to_roll = 0


    for y in range(num_rows):
        new_top_row = status_array[y,:]
        new_bottom_row = status_array[(y-1)%num_rows,:]

        boundary_slice= np.concatenate((new_top_row,new_bottom_row))

        border_count  = np.sum(boundary_slice ==  0)
        outside_count = np.sum(boundary_slice == -1)

        if border_count < min_border_count :
            min_border_count = border_count
            max_outside_count = outside_count
            pts_1_to_roll = y

            
        elif border_count == min_border_count :
            if outside_count > max_outside_count:
                max_outside_count = outside_count
                pts_1_to_roll = y
    
    ds = periodic_value_roll(ds, -pts_1_to_roll, 'Points_1')

    rolled_mask = ds.cavity_status > 0
    
    #plt.imshow(rolled_mask,cmap='bwr')
    #print(f'The grid was rolled {-pts_0_to_roll} along Points_0 and {-pts_1_to_roll} along Points_1')

    ds = force_periodicity(ds)
    return pts_0_to_roll, pts_1_to_roll    




    
def filter_data(df, threshold=threshold, file_name='df', save=False,clip=True,rolling=True,knob_Pts=[0,0]):
    correct_periodicity(df)

    ds = df.set_index(['Points_1', 'Points_0']).sort_index().to_xarray()
    mid_x = ds.sizes['Points_0']
    mid_y = ds.sizes['Points_1']
    
    ds = dataset_3x3_tiling(ds)

    # CLASSIFY
    mask = ds.hcavity > threshold
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    inner = binary_erosion(mask.values, structure=cross, border_value=1)
    ds['cavity_status'] = xr.where(mask, 0, -1)
    ds['cavity_status'] = xr.where(inner, 1, ds.cavity_status)

    #save the input df with the new column 
    ds_to_save = ds
    if clip :
        ds_to_save = ds.isel(Points_1=slice(mid_y, mid_y*2), Points_0=slice(mid_x, 2*mid_x))
    df = ds_to_save.to_dataframe().reset_index()

    # first filter : a erosion dilation to clean out scattered isolated and small part of the mask :
    raw_mask = (ds['cavity_status'] >=0)#.values
    eroded    = binary_erosion(raw_mask,  structure=cross, border_value=1)
    dilated   = binary_dilation(eroded,    structure=cross, border_value=0)
    clean_mask = dilated & raw_mask

    ds['cavity_status'] = ds['cavity_status'].where(clean_mask,-1)
    
    # 2nd filter : eliminate narrow bridges :
    ds = enforce_minimum_connection_width(ds)
    
    # pathfinding step :
    labeled, _   = label(ds.cavity_status >= 0, structure=cross)
    

    # Handling the periodic bc along xy:
    labeled_bc = labeled.copy()
    # for all y points :
    for y in range(labeled_bc.shape[0]):
        # if a group is along the 0 boundary and a different one is along the -1 boundary at the same y :
        if (labeled_bc[y,0] > 0 and labeled_bc[y,-1] > 0) and (labeled_bc[y,0] != labeled_bc[y,-1] ) :
            # The group on the -1 boundary take the value of the other group :
            labeled_bc[labeled_bc == labeled_bc[y,-1]] = labeled_bc[y,0]
    # for all x points :
    for x in range(labeled_bc.shape[1]):
        # if a group is along the 0 boundary and a different one is along the -1 boundary at the same x :
        if (labeled_bc[0,x] > 0 and labeled_bc[-1,x] > 0) and (labeled_bc[0,x] != labeled_bc[-1,x] ) :
            # The group on the -1 boundary take the value of the other group :
            labeled_bc[labeled_bc == labeled_bc[-1,x]] = labeled_bc[0,x]


    # Check Y-Axis (Top -> Midpoint -> Bottom)
    labels_start_y = np.unique(labeled_bc[0, :])
    labels_mid_y   = np.unique(labeled_bc[mid_y, :])
    labels_end_y   = np.unique(labeled_bc[-1, :])

    valid_y = np.intersect1d(labels_start_y, labels_end_y)
    valid_y = np.intersect1d(valid_y, labels_mid_y)

    # Check X-Axis (Left -> Midpoint -> Right)
    labels_start_x = np.unique(labeled_bc[:, 0])
    labels_mid_x   = np.unique(labeled_bc[:, mid_x])
    labels_end_x   = np.unique(labeled_bc[:, -1])

    valid_x = np.intersect1d(labels_start_x, labels_end_x)
    valid_x = np.intersect1d(valid_x, labels_mid_x)

    # Combine Valid Labels (Union)
    # This keeps cavities that span vertically OR horizontally
    valid_main = np.union1d(valid_y, valid_x)
    # Filter out the background 
    valid_main = valid_main[valid_main > 0]
    
    final_mask = np.isin(labeled_bc, valid_main)
    ds['cavity_status'] = ds['cavity_status'].where(final_mask, -1)


    #optimize grid position : 
    ds_to_opti = ds.isel(Points_1=slice(mid_y, 2*mid_y), Points_0=slice(mid_x, 2*mid_x))
    if rolling :
        pts_0_to_roll ,pts_1_to_roll = optimize_bc(ds_to_opti)
        ds = periodic_value_roll(ds, -pts_0_to_roll-knob_Pts[0], 'Points_0')
        ds = periodic_value_roll(ds, -pts_1_to_roll-knob_Pts[1], 'Points_1')
    
    if clip :
        ds = ds.isel(Points_1=slice(mid_y, 2*mid_y), Points_0=slice(mid_x, 2*mid_x))

    df_filtered = ds.to_dataframe().reset_index()

    if save:
        trsh_str = str(threshold).replace(".", "_")
        filtered_file_name = f'filtered_{file_name}_trsh{trsh_str}'
        df_filtered.to_csv(f'../data_filtered/{filtered_file_name}.csv', index=False)

    return df, df_filtered


    
##############################
# .STL creation : 
##############################

    
 
def choose_where_to_cut(da_chosen,boundaries,group_ids,ds,b_coords=None):

    da_chosen = xr.DataArray(da_chosen, dims=["Points_1", "Points_0"],coords={"Points_1": ds.Points_1.values, "Points_0": ds.Points_0.values})
    box_cond = 'n'
    if b_coords != None :
        fig, ax = plt.subplots()
        da_chosen.plot(ax=ax,
        cmap="tab10",          
        levels=boundaries,     # Forces discrete color blocks
        cbar_kwargs={
            "ticks": group_ids, # Puts the numbers 1, 2, 3 right next to their colors
            "label": "different cavities"
        }
    )
        
        # Rectangle takes (xy_bottom_left, width, height)
        width  = b_coords[1] - b_coords[0]
        height = b_coords[3] - b_coords[2]
        
        rect = patches.Rectangle(
            (b_coords[0], b_coords[2]), 
            width, 
            height, 
            linewidth=1, 
            edgecolor='k', 
            facecolor='grey',
            alpha=0.3
        )
        ax.add_patch(rect)
        plt.show()

        print(f"is this domain correct ? y/n ?\n{b_coords}")
        box_cond = str(input())

    if b_coords == None or box_cond == 'n' :
        
        while box_cond == 'n' :
            print('enter the desired domain coordinates : x_min x_max y_min y_max] :')
            b_coords = list(map(float, input("Enter numbers separated by space: ").split()))
            # Plot the DataArray and capture the axis (ax)
            fig, ax = plt.subplots()
            da_chosen.plot(ax=ax)
            
            # Rectangle takes (xy_bottom_left, width, height)
            width  = b_coords[1] - b_coords[0]
            height = b_coords[3] - b_coords[2]
            
            rect = patches.Rectangle(
                (b_coords[0], b_coords[2]), 
                width, 
                height, 
                linewidth=1, 
                edgecolor='k', 
                facecolor='grey',
                alpha=0.3
            )
            ax.add_patch(rect)
            plt.show()
            
            print("is this domain correct ? y/n ? ")
            box_cond = str(input())
    return b_coords
            
def get_snappy_paint_buckets(ds,b_coords=None, choose_1_body=False):
    """
    Finds a safe internal (X, Y, Z) coordinate for every disconnected 
    fluid body in the domain to use as locationsInMesh in snappyHexMesh.
    """
    fluid_mask = (ds.cavity_status.values >= 0)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_bodies, num_bodies = label(fluid_mask, structure=cross)

    da = xr.DataArray(labeled_bodies, dims=["Points_1", "Points_0"],coords={"Points_1": ds.Points_1.values, "Points_0": ds.Points_0.values})
    
    
    da_groups = da.where(da > 0)

    boundaries = np.arange(0.5, num_bodies + 1.5, 1)
    group_ids = np.unique(labeled_bodies)
    
    if choose_1_body :
        da_groups.plot(
            cmap="tab10",          
            levels=boundaries,     # Forces discrete color blocks
            cbar_kwargs={
                "ticks": group_ids, # Puts the numbers 1, 2, 3 right next to their colors
                "label": "different cavities"
            }
        )
        plt.axvline(13.0,color='k',linestyle='--')
        plt.axhline(13.0,color='k',linestyle='--')
        plt.axvline(26.0,color='k',linestyle='--')
        plt.axhline(26.0,color='k',linestyle='--')
        plt.show()

        print(f'which group should we keep to mesh ?\n')
        group=int(input())
        print(f'group selected is {group} :')
    
        da_chosen = da.where(da == group)
    else :
        da_chosen = da_groups
    box_coords = choose_where_to_cut(da_chosen,boundaries,group_ids,ds,b_coords = b_coords)
    
    print("\n--- Calculating snappyHexMesh locationsInMesh ---\n")
    ds_slice = ds.sel(Points_1=slice(box_coords[2], box_coords[3]), Points_0=slice(box_coords[0], box_coords[1])) # inside the final domain
    da_slice = da_chosen.sel(Points_1=slice(box_coords[2], box_coords[3]), Points_0=slice(box_coords[0], box_coords[1])) # inside the final domain
    
    # Get the binary mask of inner valid fluid cells
    fluid_mask = (da_slice > 0)
    
    # Label the disconnected bodies (no periodic wrapping)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled_bodies, num_bodies = label(fluid_mask, structure=cross)
    
    locations = []
    da_slice.plot(
        cmap="tab10",          
        levels=boundaries,     # Forces discrete color blocks
        cbar_kwargs={
            "ticks": group_ids, # Puts the numbers 1, 2, 3 right next to their colors
            "label": "different cavities"
        }
    )
    for i in range(1, num_bodies + 1):
        # Get all Y, X indices that belong to this specific body
        y_indices, x_indices = np.where(labeled_bodies == i)

        z_min = ds['bedrock'].values.min()
        z_max = ds['zb'].values.max()
        print(f'On the domain the cavity span from z_min={z_min:.2f},to z_max={z_max:.2f}')
        
        median_idx = len(x_indices) // 2
        safe_x_idx = x_indices[median_idx]
        safe_y_idx = y_indices[median_idx]
        
        # Extract the true physical coordinates
        x_coord = ds_slice.Points_0.values[safe_x_idx]
        y_coord = ds_slice.Points_1.values[safe_y_idx]
        
        # Extract floor and ceiling
        z_bed = ds_slice.bedrock.values[safe_y_idx, safe_x_idx]
        z_top = ds_slice.zb.values[safe_y_idx, safe_x_idx]
        z_coord = z_bed + (z_top - z_bed) / 2.0
        
        # Format it exactly as OpenFOAM expects it
        foam_string = f"    (( {x_coord:.4f}  {y_coord:.4f}  {z_coord:.4f} ) cavity)"
        locations.append(foam_string)
        
        print(f"Found Body {i}: {len(y_indices)} pixels -> Safe point at X:{x_coord:.2f}, Y:{y_coord:.2f}, Z:{z_coord:.2f}")
    
        plt.scatter(x_coord,y_coord,color='m',marker='x')
    plt.show()

    # Print the formatted block ready to be copy/pasted : 

    print(f'coords are : {box_coords}\n')
    print("\nCopy this block into your blockMeshDict:\n")
    print("vertices")
    print("(")
    print(f'    ( {box_coords[0]} {box_coords[2]} -2.5)  //0') # x_min, y_min, z_min
    print(f'    ( {box_coords[1]} {box_coords[2]} -2.5)  //1') # x_max, y_min, z_min
    print(f'    ( {box_coords[1]} {box_coords[3]} -2.5)  //2') # x_max, y_max, z_min
    print(f'    ( {box_coords[0]} {box_coords[3]} -2.5)  //3') # x_min, y_max, z_min
    print(f'    ( {box_coords[0]} {box_coords[2]}  2.5)  //4') # x_min, y_min, z_max
    print(f'    ( {box_coords[1]} {box_coords[2]}  2.5)  //5') # x_max, y_min, z_max
    print(f'    ( {box_coords[1]} {box_coords[3]}  2.5)  //6') # x_max, y_max, z_max
    print(f'    ( {box_coords[0]} {box_coords[3]}  2.5)  //7') # x_min, y_max, z_max
    print(');\n')

    X_res_blockMesh = (box_coords[1] - box_coords[0])/0.2
    Y_res_blockMesh = (box_coords[3] - box_coords[2])/0.2
    
    print('blocks')
    print('(')
    print(f'    hex (0 1 2 3 4 5 6 7) ({X_res_blockMesh:.0f} {Y_res_blockMesh:.0f} 25) simpleGrading (1 1 1)')
    print(');')
    print("-------------------------------------------------\n")

    
    print("\nCopy this block into your snappyHexMeshDict:\n")
    print("locationsInMesh")
    print("(")
    for loc in locations:
        print(loc)
    print(");")
    print("-------------------------------------------------\n")

    ds_chosen = ds.copy(deep=True)
    
    # Keep cavity_status where our chosen body is, but force it to -1 everywhere else.
    valid_body_mask = ~np.isnan(da_chosen)
    ds_chosen['cavity_status'] = xr.where(valid_body_mask, ds['cavity_status'], -1)
    
    return box_coords, ds_chosen




def create_surface_mesh(df,b_coords=None, save=False, file_name="cavity_surface", threshold=threshold,choose_1_body=False):
    
    ds = df.set_index(['Points_1', 'Points_0']).sort_index().to_xarray()
    box_coords, ds_chosen = get_snappy_paint_buckets(ds, b_coords=b_coords,choose_1_body=choose_1_body)
    
    # Raw data extraction
    status_raw = ds_chosen['cavity_status'].values
    zb_raw = ds_chosen['zb'].values
    bedrock_raw = ds_chosen['bedrock'].values
    
    facteur_zoom = 10
    print(f' -- oversampling -- ')
    
    # Oversampling & smoothing to ensure continuity
    zb_high_res = zoom(zb_raw, zoom=facteur_zoom, order=3, mode='wrap')
    bedrock_high_res = zoom(bedrock_raw, zoom=facteur_zoom, order=3, mode='wrap')
    zb_smooth = gaussian_filter(zb_high_res, sigma=1.0, mode='wrap')
    bedrock_smooth = gaussian_filter(bedrock_high_res, sigma=1.0, mode='wrap')

    status_high_res = np.round(zoom(status_raw, zoom=facteur_zoom, order=1, mode='wrap')).astype(int)
    
    # Recompute hcavity (This is our continuous scalar field for smooth clipping)
    true_height = zb_smooth - bedrock_smooth

    # If it was tagged as a wall (0), but has clearance, force it to be inside (1)
    status_high_res[(status_high_res == 0) & (true_height > threshold)] = 1

    # Boolean mask: True if considered a cavity
    inside_mask = (status_high_res >= 0)
    
    # Erode slightly the cavity
    cross = [[0,1,0],[1,1,1],[0,1,0]]
    status_high_res_eroded = binary_erosion(inside_mask, structure=cross, mask=inside_mask, border_value=1)
    status_high_res = status_high_res_eroded - 1 
    inside_mask = (status_high_res >= 0)
    
    # Create a taper of zb toward bedrock
    distance_map = distance_transform_edt(inside_mask)

    distance_map = gaussian_filter(distance_map, sigma=1.5, mode='wrap')
    
    pixels_to_taper = 8.0 
    
    # Normalize to create a multiplier between 0.0 and 1.0
    linear_taper = np.clip(distance_map / pixels_to_taper, 0.0, 1.0)
    
    # The Concave Taper (Quadratic Parabola)
    # - At the interior (1.0): slope is 0 (continuous and flat against the roof)
    # - At the bedrock (0.0): meets the floor with a scooped, concave curve
    smooth_taper = np.sqrt(1.0 - (1.0 - linear_taper)**2)
    
    cavity_height_curved = true_height * smooth_taper
    zb_tapered = bedrock_smooth + cavity_height_curved

    print(f' -- building the 3D grid  -- \n')
    
    x_min, x_max = ds_chosen.Points_0.min().item(), ds_chosen.Points_0.max().item()
    y_min, y_max = ds_chosen.Points_1.min().item(), ds_chosen.Points_1.max().item()
    
    x_new = np.linspace(x_min, x_max, zb_smooth.shape[1])
    y_new = np.linspace(y_min, y_max, zb_smooth.shape[0])
    

    xx, yy = np.meshgrid(x_new, y_new)

    X_3d = np.stack([xx, xx], axis=-1)
    Y_3d = np.stack([yy, yy], axis=-1)
    Z_3d = np.stack([bedrock_smooth, zb_tapered], axis=-1)
    
    grid_3d = pv.StructuredGrid(X_3d, Y_3d, Z_3d)

    #Map the continuous distance field to the 3D grid
    
    dist_3d = np.stack([distance_map, distance_map], axis=-1)
    grid_3d.point_data["distance"] = dist_3d.flatten(order="F")

    print(f' -- Extracting smooth contour surface -- \n')
    
    # Clip at distance = 0.6 to get a continuous outer perimeter that respects cavity_status
    clipped_grid = grid_3d.clip_scalar(scalars="distance", value=0.6, invert=False)
    soufflerie_brute = clipped_grid.extract_surface(algorithm='dataset_surface').clean()
    
    surface_3x3 = soufflerie_brute.clean()

    #split all the different bodies :
    bodies = surface_3x3.split_bodies()
    print(f'There are {len(bodies)} different bodies to handle')

    processed_bodies = []

    AOI_x_min = box_coords[0]
    AOI_x_max = box_coords[1]
    AOI_y_min = box_coords[2]
    AOI_y_max = box_coords[3]

    save_counter = 0
    for i, s_body in enumerate(bodies) :
        
        print(f'Processing body{i+1}/{len(bodies)}')
        
        body = s_body.extract_surface(algorithm='dataset_surface').triangulate()
        
        AOI_z_min = body.bounds[4]
        AOI_z_max = body.bounds[5]

        AOI_box_bounds = [AOI_x_min, AOI_x_max, AOI_y_min, AOI_y_max, AOI_z_min, AOI_z_max]

        AOI_surface_walls = body.clip_box(AOI_box_bounds, invert=False) 

        # check if the surface we're treating is present in the AoI, go to the next body if not
        if AOI_surface_walls.is_empty :
            print(f'-- body not in AoI  --')
            continue
        
        AOI_surface_walls = AOI_surface_walls.extract_surface(algorithm="dataset_surface").triangulate().clean(tolerance=1e-3)

        
         # export 
        if save :
            print(f'-- saving body --')
            path_name=f'../2D_mesh/{file_name}_body_{save_counter}_2D'
            body.save(path_name+'_walls.stl', binary=True)
            save_counter+=1

            
        processed_bodies.append({'surface_walls': body,
                                 'AOI_surface_walls'  : AOI_surface_walls})

        
    return processed_bodies, box_coords, ds_chosen





###########################################
# Simulation setup & post processing
###########################################






def initialisation():
    rho_w             = 1000 # kg/m3
    rho_i             = 920  # kg/m3
    H                 = 100  # m
    g                 = 9.81 # m/s2
    overburden_ratio  = 0.7  # adim
    basal_slope       = 5    # degrees°
    domain_length_L   = 13   # m

    P_R_channel       = 0    # Pa
    dist_to_R_channel = 100  # m

    basal_slope = np.deg2rad(basal_slope)
    
    P_i     = rho_i * g * H           # Pa
    P_inlet = overburden_ratio * P_i  # Pa
    
    delta_z = domain_length_L * np.sin(basal_slope)

    delta_P_w_min = 0 # Pa
    delta_phi_min = rho_w*g*delta_z + delta_P_w_min  # Pa
    grad_eff_phi_min = delta_phi_min/domain_length_L # Pa/m
    
    # the water pressure gradient between the inlet and the R-channel :
    grad_P_w = (P_inlet-P_R_channel)/dist_to_R_channel
    
    delta_P_w_max = grad_P_w * domain_length_L

    delta_phi_max = rho_w*g*delta_z + delta_P_w_max
    grad_eff_phi_max = delta_phi_max/domain_length_L
    
    grad_eff_phi_list = np.linspace(grad_eff_phi_min,grad_eff_phi_max,10)
    
    # we add 3 more extreme values to the list : 
    # 2 low grad :
    delta_P_w= np.array([1.5,5.5])
    delta_phi = rho_w*g*delta_z + delta_P_w*rho_w  # Pa
    grad_eff_phi = delta_P_w*1000/domain_length_L # Pa/m
    # 1 extreme grad :
    tmp_float=(rho_w*g*delta_z + (631.76-500.0)*rho_w)/domain_length_L 
    # gymnastic to append each value on the appropriate side :
    tmp = np.append(grad_eff_phi_list,tmp_float)
    grad_eff_phi_list=np.append(grad_eff_phi , tmp)

    
    kinematic_P_outlet_list  = [] # m2/s2
    print(f'P_inlet/rho_w = {(P_inlet)/rho_w:.2f} m2/s2\n')
    for grad_eff_phi in grad_eff_phi_list :
        
        P_outlet=(P_inlet-grad_eff_phi*domain_length_L)
        kinematic_P_outlet_list.append(P_outlet/rho_w)
        print(f'  P_outlet/rho_w = {(P_outlet)/rho_w:.2f}  m2/s2\n    grad_eff_phi = {grad_eff_phi:.2f} Pa/m \n\n')

    return kinematic_P_outlet_list, grad_eff_phi_list



def get_boundaries_specs(body_dic,box_coords):
    # 
    boundary_lines = body_dic[0]['AOI_surface_walls'].extract_feature_edges(
                                                        boundary_edges=True,
                                                        feature_edges=False,
                                                        manifold_edges=False,
                                                        non_manifold_edges=False)
    clip_x_min = box_coords[0]
    clip_x_max = box_coords[1]
    clip_y_min = box_coords[2]
    clip_y_max = box_coords[3]
    
    separated_lines = boundary_lines.split_bodies()
    mean_coords = {}
    
    boundaries_values = {'x_max' :{ 'area' : None , 'perimeter' : None},
                         'x_min' :{ 'area' : None , 'perimeter' : None},
                         'y_max' :{ 'area' : None , 'perimeter' : None},
                         'y_min' :{ 'area' : None , 'perimeter' : None}}
    
    area_x_max = 0
    perimeter_x_max = 0
    area_x_min = 0
    perimeter_x_min = 0
    area_y_max = 0
    perimeter_y_max = 0
    area_y_min = 0
    perimeter_y_min = 0
    
    for i, single_line in enumerate(separated_lines):
        # .points returns an (N, 3) numpy array of [X, Y, Z] coordinates
        points = single_line.points
        
        # points[:, 0] is X, points[:, 1] is Y
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        
        mean_coords[f"Line_{i}"] = (mean_x, mean_y)
        
        area = 0 
        perimeter = 0
        tol=1e-2
        if np.isclose(mean_x,clip_x_max,atol=tol):
            area = single_line.delaunay_2d().area
            perimeter = single_line.length
            area_x_max += area
            perimeter_x_max += perimeter
            boundaries_values['x_max']={ 'area' : area_x_max , 'perimeter' : perimeter_x_max}
            
        elif np.isclose(mean_x,clip_x_min,atol=tol):
            area = single_line.delaunay_2d().area
            perimeter = single_line.length
            area_x_min += area
            perimeter_x_min += perimeter
            boundaries_values['x_min']={ 'area' : area_x_min , 'perimeter' : perimeter_x_min}
            
        elif np.isclose(mean_y,clip_y_max,atol=tol):
            area = single_line.delaunay_2d().area
            perimeter = single_line.length
            area_y_max += area
            perimeter_y_max += perimeter
            boundaries_values['y_max']={ 'area' : area_y_max , 'perimeter' : perimeter_y_max}
            
        elif np.isclose(mean_y,clip_y_min,atol=tol):
            area = single_line.delaunay_2d().area
            perimeter = single_line.length
            area_y_min += area
            perimeter_y_min += perimeter
            boundaries_values['y_min']={ 'area' : area_y_min , 'perimeter' : perimeter_y_min}
            
        else :
            print(f'line {i} is on 2 faces of the bounding box and needs further handling')
        
        print(f"Line {i}: Mean X = {mean_x:.3f}, Mean Y = {mean_y:.3f}")
    return boundaries_values



def turbulent_initialisation(body_dic,box_coords,inlet='x_min'):
    boundaries_values=get_boundaries_specs(body_dic,box_coords)
    area = boundaries_values[inlet]['area']
    wetted_perimeter = boundaries_values[inlet]['perimeter']
    I       = 0.02 # adim 
    u_inlet = 0.01 # m/s
    C_nu    = 0.09 # adim
    
    D_h    = 4* area / wetted_perimeter 
    k      = 1.5 * (u_inlet * I )**2
    omega  = k**0.5 / (C_nu**0.25 * 0.07 * D_h)

    print(f'k = {k:.4e}, omega = {omega:.4e}')
    
    return  boundaries_values,k, omega






def get_sim_data(directory_name,ovb_ratio,trsh_val,box_coords,save=True,mean_val=False,yPlus=False,res=0.05,filename_save='test'):
    post_process_file="/home/grandafe/OpenFOAM/grandafe-v2412/run/"+ovb_ratio+"/trsh_"+trsh_val+"/"+directory_name+"/case.foam"
    
    reader = pv.OpenFOAMReader(post_process_file)
    reader.cell_to_point_creation = False
    reader.enable_all_cell_arrays()
    
    # Optional: Choose the time step. Defaults to the latest time step.
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    
    # OpenFOAM cases load as a MultiBlock dataset; we want the internal mesh
    internal_mesh = mesh["internalMesh"]
    internal_mesh_points_only = internal_mesh.cell_data_to_point_data(pass_cell_data=False)
    
    
    # 2. Define your desired 3D grid resolution
    nx, ny, nz = int(( box_coords[1] - box_coords[0])//res), int(( box_coords[3] - box_coords[2])//res), 50 
    bounds = internal_mesh_points_only.bounds
    x = np.linspace(box_coords[0], box_coords[1], nx)
    y = np.linspace(box_coords[2], box_coords[3], ny)
    z = np.linspace(bounds[4]-0.1, bounds[5]+0.1, nz)
    
    grid = pv.RectilinearGrid(x, y, z)
    
    # 3. Resample the unconflicted point data onto the grid
    resampled = grid.sample(internal_mesh_points_only)
        
        
    # 4. Extract points and fields safely from point_data
    points = resampled.points  
    
    U = np.linalg.norm(resampled.point_data["U"], axis=1)
    U_y = resampled.point_data["U"][:, 1]
    p = resampled.point_data["p"] 
    k = resampled.point_data['k']
    omega = resampled.point_data['omega']

    if mean_val:
        kMean = resampled.point_data["kMean"] 
        pMean = resampled.point_data["pMean"]
        omegaMean = resampled.point_data['omegaMean']

        UMean_vec = resampled.point_data["UMean"]
        UMean= np.linalg.norm(UMean_vec, axis=1)

        UMean_x = UMean_vec[:, 0]
        UMean_y = UMean_vec[:, 1]
        UMean_z = UMean_vec[:, 2]
        
        UPrime2Mean = np.linalg.norm(resampled.point_data["UPrime2Mean"], axis=1)
    if yPlus:    
        yPlus = resampled.point_data["yPlus"] 
    
    grid_shape = (nz, ny, nx)

    u_3d = U.reshape(grid_shape)
    uy_3d = U_y.reshape(grid_shape)
    p_3d = p.reshape(grid_shape)
    k_3d = k.reshape(grid_shape)
    omega_3d = omega.reshape(grid_shape)
    
    if mean_val:
        u_mean_3d  = UMean.reshape(grid_shape)

        ux_mean_3d  = UMean_x.reshape(grid_shape)
        uy_mean_3d  = UMean_y.reshape(grid_shape)
        uz_mean_3d  = UMean_z.reshape(grid_shape)
        
        u_prime_3d = UPrime2Mean.reshape(grid_shape)
        p_mean_3d  = pMean.reshape(grid_shape)
        k_mean_3d  = kMean.reshape(grid_shape)
        omega_mean_3d  = omegaMean.reshape(grid_shape)
    if yPlus:
        yPlus_3d   = yPlus.reshape(grid_shape)
    
    # 6. Pack into an Xarray Dataset

    ds = xr.Dataset(
        data_vars={
            "U": (["z", "y", "x"], u_3d),
            "U_y": (["z", "y", "x"], uy_3d),
            "p": (["z", "y", "x"], p_3d),
            "k": (["z", "y", "x"], k_3d),
            "omega": (["z", "y", "x"], omega_3d),
        },
        coords={
            "z": z,
            "y": y,
            "x": x,
        },  
    )
    
    if mean_val:
        ds = xr.Dataset(
            data_vars={
                "U": (["z", "y", "x"], u_3d),
                "U_y": (["z", "y", "x"], uy_3d),
                "p": (["z", "y", "x"], p_3d),
                "k": (["z", "y", "x"], k_3d),
                "omega": (["z", "y", "x"], omega_3d),
                "U_mean": (["z", "y", "x"], u_mean_3d),
                "Ux_mean": (["z", "y", "x"], ux_mean_3d),
                "Uy_mean": (["z", "y", "x"], uy_mean_3d),
                "Uz_mean": (["z", "y", "x"], uz_mean_3d),
                "U_Prime2Mean": (["z", "y", "x"], u_prime_3d),
                "p_mean": (["z", "y", "x"], p_mean_3d),
                "k_mean": (["z", "y", "x"], k_mean_3d),
                "omega_mean": (["z", "y", "x"], omega_mean_3d),
            },
            coords={
                "z": z,
                "y": y,
                "x": x,
            },  
        )
        
        ds_masked = ds.where(np.abs(ds["p_mean"]) >= 1e-6)
    else :
        ds_masked = ds.where(np.abs(ds["p"]) >= 1e-6)
        
    # 2. Collapse the Z-axis (ignoring the NaNs we just created)
    ds_2d = ds_masked.mean(dim="z")


    
    
    if save :
        df_export = ds_masked.to_dataframe().reset_index()
        df_export.to_csv(f'/home/grandafe/Documents/Stage/post_sim_data/{ovb_ratio}/trsh_{trsh_val}/{filename_save}.csv')

        
    return ds_masked, ds_2d, internal_mesh







def merge_openfoam_dat(pathto_file1, pathto_file2, pathto_output_file):
    # Check if input files exist
    if not os.path.exists(pathto_file1):
        print(f"Error: Could not find {pathto_file1}")
        return
    if not os.path.exists(pathto_file2):
        print(f"Error: Could not find {pathto_file2}")
        return

    with open(pathto_output_file, 'w') as outfile:
        # 1. Write the first file completely (including its header)
        with open(pathto_file1, 'r') as f1:
            for line in f1:
                outfile.write(line)
        
        # 2. Write the second file, but skip the comment/header lines
        with open(pathto_file2, 'r') as f2:
            for line in f2:
                if not line.startswith('#'):
                    outfile.write(line)
        
    return 



    

def get_flow_rates(directory_name,ovb_ratio,trsh_val,kP_out_str,start_Time='3000',merge = False, files_to_merge=[3000,4500]):
    
    flowrate_dir = f"/home/grandafe/OpenFOAM/grandafe-v2412/run/{ovb_ratio}/trsh_{trsh_val}/{directory_name}/postProcessing/outletFlowRate/{start_Time}/surfaceFieldValue.dat"
    
    if merge :
        flowrate_path  = f"/home/grandafe/OpenFOAM/grandafe-v2412/run/{ovb_ratio}/trsh_{trsh_val}/{directory_name}/postProcessing/outletFlowRate/"
        flowrate_dir_1 = f"{flowrate_path}/{files_to_merge[0]}/surfaceFieldValue.dat"
        flowrate_dir_2 = f"{flowrate_path}/{files_to_merge[1]}/surfaceFieldValue.dat"
        output_dir     = f'{flowrate_path}/merged_surfaceFieldValue.dat'
        merge_openfoam_dat(flowrate_dir_1,flowrate_dir_2,output_dir)
        flowrate_dir = output_dir
        
    flowrate_df=pd.read_csv(
        flowrate_dir, 
        sep=r'\s+',          # Regex separator to handle variable whitespace
        skiprows=4,          # Skip the 4 comment lines
        header=0,            # Use the 5th line as the column header
        comment='#'          # Just in case there are other comments
    )
    
    # Rename columns for easier access (optional)
    flowrate_df.columns = ['Time', f'FlowRate{kP_out_str}']
    return flowrate_df




######################
# get the min cross-section a 
######################




def find_2d_constriction_by_dilation(ds_2d_p, start_p=50.0, stop_p=0.0, periodic_Y=True):
    """
    Grows a high-pressure region until it hits an under-pressure threshold.
    Returns the region mask and the frontier (the constriction interface).
    """
    # Extract the 2D numpy array and create masks
    p_array = ds_2d_p.values
    valid_cavity = ~np.isnan(p_array)
    
    # Safe array for comparisons (converts NaN to a dummy negative value)
    safe_p = np.nan_to_num(p_array, nan=-999)
    
    # Define the regions
    current_region = valid_cavity & (safe_p >= start_p)
    allowed_growth_zone = valid_cavity & (safe_p > stop_p)
    
    if not current_region.any():
        print(f"Error: No starting region found with pressure >= {start_p}%")
        return None, None

    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    
    iteration = 0
    max_iter = 10000 
    
    # The Masked Region Growing Loop
    while iteration < max_iter:
        
        if periodic_Y:
            # Periodic Y-axis Padding
            padded_region = np.pad(current_region, pad_width=((1, 1), (0, 0)), mode='wrap')
            dilated_padded = binary_dilation(padded_region, structure=structure)
            dilated = dilated_padded[1:-1, :]
        else:
            # Normal dilation WITHOUT periodic wrap-around
            dilated = binary_dilation(current_region, structure=structure)
            
        # The region can only grow if the new pixels are > stop_p (0%)
        dilated = dilated & allowed_growth_zone
        
        # Check if the region grew at all
        frontier = dilated ^ current_region
        
        if not frontier.any():
            print(f"Flood stopped growing at iteration {iteration}. Constriction reached!")
            break
            
        current_region = dilated
        iteration += 1

    # Extract the exact constriction interface 
    if periodic_Y:
        padded_final = np.pad(current_region, pad_width=((1, 1), (0, 0)), mode='wrap')
        dilated_final = binary_dilation(padded_final, structure=structure)[1:-1, :]
    else:
        dilated_final = binary_dilation(current_region, structure=structure)
    
    # The interface is the boundary of the final region, constrained to the actual cavity
    constriction_interface = (dilated_final ^ current_region) & valid_cavity

    return current_region, constriction_interface





def get_plane_from_interface(ds_2d_p, constriction_interface):
    """
    Calculates the best-fit origin and normal vector for a boolean interface mask.
    """
    # Get the array indices where the interface is True
    y_idx, x_idx = np.where(constriction_interface)
    
    if len(x_idx) == 0:
        print("No interface found to fit.")
        return None, None
        
    # Convert indices to physical X and Y coordinates
    x_coords = ds_2d_p.x.values[x_idx]
    y_coords = ds_2d_p.y.values[y_idx]
    
    # Handle Y-Axis Periodicity for the Centroid
    # If the points span across the top and bottom edges, an average will incorrectly 
    # place the centroid in the middle of the domain. We must unwrap them.
    y_min = ds_2d_p.y.min().values
    y_max = ds_2d_p.y.max().values
    y_domain_length = y_max - y_min
    
    # If the spread in Y is greater than half the domain, it's wrapping around
    if (y_coords.max() - y_coords.min()) > (y_domain_length / 2.0):
        # Shift points in the top half down by one full domain length
        y_coords = np.where(y_coords > (y_min + y_domain_length / 2.0), 
                            y_coords - y_domain_length, 
                            y_coords)
                            
    # Calculate the Origin (Centroid)
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    # Wrap centroid_y back into the normal domain bounds if we shifted it
    if centroid_y < y_min:
        centroid_y += y_domain_length
        
    # Calculate the Normal Vector using PCA
    # Stack the points into a 2D matrix
    points = np.vstack((x_coords, y_coords)).T
    
    # Mean-center the points
    centered_points = points - [np.mean(x_coords), np.mean(y_coords)]
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # The normal vector is the eigenvector associated with the SMALLEST eigenvalue 
    # (the direction of least variance, perpendicular to the interface)
    normal_2d = eigenvectors[:, 0]
    

        
    # Format for PyVista 3D slicing
    origin = [centroid_x, centroid_y, 0.0]
    normal = [normal_2d[0], normal_2d[1], 0.0]
    
    return origin, normal


def fill_concave_wireframe(wireframe_slice):
    """
    Uses VTK's contour triangulator to fill a 1D loop with a solid 2D surface.
    (Unlike Delaunay, this respects concave boundaries.)
    """
    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(wireframe_slice)
    triangulator.Update()
    return pv.wrap(triangulator.GetOutput())






def find_minimum_area_from_stl(stl_file_name, seed_origin, normal, sweep_dist=0.05, num_planes=20, box_size=1.0):
    """
    Sweeps parallel planes along a normal vector to find the absolute minimum geometric area.
    
    Parameters:
    - original_mesh: The full PyVista unstructured grid.
    - seed_origin: [X, Y, Z] coordinate from the 2D pressure algorithm.
    - normal: [X, Y, Z] normal vector from the PCA algorithm.
    - sweep_dist: The distance (in meters) to sweep forward and backward from the seed.
    - num_planes: How many parallel slices to take.
    - box_size: The radius (in meters) of the bounding box to clip before slicing.
    """
    
    stl_file_path = f'/home/grandafe/Documents/Stage/2D_mesh/{stl_file_name}_body_0_2D_walls.stl'
    # Load the .stl surface geometry
    stl_mesh = pv.read(stl_file_path)
    
    seed_origin = np.array(seed_origin)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal) 
    
    # Box Clip the STL to make sure we're evaluating only the constriction
    bounds = [
        seed_origin[0] - box_size, seed_origin[0] + box_size,
        seed_origin[1] - box_size, seed_origin[1] + box_size,
        seed_origin[2] - box_size, seed_origin[2] + box_size
    ]
    
    local_mesh = stl_mesh.clip_box(bounds, invert=False)
    
    if local_mesh.n_cells == 0:
        print("Error: Bounding box clip is empty.")
        return None
        
    distances = np.linspace(0, sweep_dist, num_planes)
    
    min_area = float('inf')
    best_origin = None
    best_slice_body = None
    
    # Perform a forward sweep on the STL to get the best min area possible
    for d in distances:
        current_origin = seed_origin + (d * normal)
        
        # Slicing an STL gives a hollow 1D wireframe ring
        wireframe_slice = local_mesh.slice(origin=current_origin, normal=normal)
        
        if wireframe_slice.n_cells > 0:
            try:
                # This explicitly fills the inside of the lines, respecting concavity
                solid_slice = fill_concave_wireframe(wireframe_slice)

                
                
            except Exception as e:
                print(f"Skipping a plane due to triangulator error: {e}")
                continue
            target_area = solid_slice.area
            sized_wireframe = wireframe_slice.compute_cell_sizes(length=True, area=False, volume=False)
            
            current_perimeter = sized_wireframe.cell_data['Length'].sum()
            """ 
            # Split bodies in case the slice caught multiple cavities
            bodies = solid_slice.split_bodies()
            
            target_area = 0.0
            min_dist = float('inf')
            target_body = None
            
            for body in bodies:
                # Find the triangulated surface closest to our flow path
                dist = np.linalg.norm(np.array(body.center) - current_origin)
                if dist < min_dist:
                    min_dist = dist
                    target_area = body.area
                    target_body = body
             """         
            if 0 < target_area < min_area:
                min_area = target_area
                best_perimeter = current_perimeter
                best_origin = current_origin
                best_slice_body = solid_slice #target_body
             
            
    return {
        'min_area': min_area,
        'best_origin': best_origin,
        'slice_mesh': best_slice_body,
        'local_domain': local_mesh,
        'min_perimeter': best_perimeter
    }





    
def visualize_constriction(results):
    if not results or results.get('slice_mesh') is None:
        print("Error: No results to visualize.")
        return

    # Initialize the plotter
    plotter = pv.Plotter()

    # 1. Plot the Local Clipped Domain
    # We use a low opacity so you can see inside, and show the mesh edges for structural context
    plotter.add_mesh(results['local_domain'], 
                     color='lightblue', 
                     opacity=0.3, 
                     show_edges=True, 
                     edge_color='gray',
                     label='Local Clipped Mesh')

    # 2. Plot the Minimum Area Slice
    # Rendered solid and bright red to stand out inside the domain
    plotter.add_mesh(results['slice_mesh'], 
                     color='red', 
                     opacity=1.0, 
                     label='Minimum Area Constriction')

    # 3. Mark the Geometric Center of the slice
    plotter.add_points(np.array([results['best_origin']]), 
                       color='black', 
                       point_size=12, 
                       render_points_as_spheres=True,
                       label='Geometric Center')

    # Add environment details
    plotter.add_axes()
    plotter.add_legend()
    plotter.set_background('white')

    # Display the interactive window
    plotter.show()





######################
# visualisation :
######################



   
def show_topview(df,value='hcavity',color='gnuplot',contour=False,logscale=False):
    pivoted = df.pivot(index='Points_1', columns='Points_0', values=value)
    if contour :
        pivoted_status = df.pivot(index='Points_1', columns='Points_0', values='cavity_status')

    # Get unique sorted values for the new meshgrid
    u_x = np.sort(df['Points_0'].unique()) 
    u_y = np.sort(df['Points_1'].unique()) 

    # Meshgrid order is (Horizontal_array, Vertical_array)
    X_new, Y_new = np.meshgrid(u_x, u_y,indexing='xy')

    plt.figure(figsize=(10, 8))
    
    # Extent is [left, right, bottom, top] 
    if value == 'cavity_status':
        im = plt.imshow(pivoted.values, 
                       extent=[u_x.min(), u_x.max(), u_y.min(), u_y.max()], 
                       origin='lower', cmap=color, vmin=-1.0, vmax=1.0)
    else :
        if logscale :
            im = plt.imshow(pivoted.values, 
                           extent=[u_x.min(), u_x.max(), u_y.min(), u_y.max()], 
                           origin='lower', cmap=color, norm=LogNorm(1e-4,10))
        else :
            im = plt.imshow(pivoted.values, 
                               extent=[u_x.min(), u_x.max(), u_y.min(), u_y.max()], 
                               origin='lower', cmap=color)
    
    if value == 'hcavity':
        lab = 'Cavity Height (m)'
    else : 
        lab = value
    
    if contour==True:
        #The outer boundary
        plt.contour(X_new, Y_new, pivoted_status.values, 
            levels=[-0.5], colors='red', linewidths=2)

        # The inner boundary 
        plt.contour(X_new, Y_new, pivoted_status.values, 
                    levels=[0.5], colors='yellow', linestyles='dashed')

    plt.colorbar(im, label=lab)

    plt.axvline(13.0,color='k',linestyle='--')
    plt.axhline(13.0,color='k',linestyle='--')
    plt.axvline(26.0,color='k',linestyle='--')
    plt.axhline(26.0,color='k',linestyle='--')

    
    plt.xlabel('X coordinate (Points_0)')
    plt.ylabel('Y coordinate (Points_1)')
    plt.title(f'topview of {lab}')
    plt.show()
    return