import numpy as np
import pandas as pd
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.signal import savgol_filter


# ------------------ Data Augmentation ------------------ #
def flip_data(data, flip_axis='horizontal'):
    """
    Flip skeleton horizontally (around X midpoint) or vertically (around Y midpoint).
    By default, this flips horizontally.
    """
    flipped_data = data.copy()
    mvmt_cols_X = [col for col in data.columns if '_X' in col]
    mvmt_cols_Y = [col for col in data.columns if '_Y' in col]

    if flip_axis == 'horizontal':
        # Flip only the X coordinates around their midpoint
        x_min = flipped_data[mvmt_cols_X].min().min()
        x_max = flipped_data[mvmt_cols_X].max().max()
        mid_x = 0.5 * (x_min + x_max)
        flipped_data[mvmt_cols_X] = 2 * mid_x - flipped_data[mvmt_cols_X]
    elif flip_axis == 'vertical':
        # Flip only the Y coordinates around their midpoint
        y_min = flipped_data[mvmt_cols_Y].min().min()
        y_max = flipped_data[mvmt_cols_Y].max().max()
        mid_y = 0.5 * (y_min + y_max)
        flipped_data[mvmt_cols_Y] = 2 * mid_y - flipped_data[mvmt_cols_Y]
    else:
        print("flip_axis must be 'horizontal' or 'vertical'; returning unmodified data.")

    return flipped_data

def add_noise(data, noise_level=0.1):
    """
    Add Gaussian noise to data. Typically done after you have
    an initial clean data. Adjust noise_level as needed.
    """
    noisy_data = data.copy()
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data += noise
    return noisy_data


# ------------------ Data Preprocessing ------------------ #
def impute_missing_values(data):
    """
    Impute missing values using IterativeImputer.
    """
    imputer = IterativeImputer(
        max_iter=100, 
        random_state=42,
        sample_posterior=False,
        skip_complete=True
    )
    cols = data.columns
    idx = data.index

    data_imputed = imputer.fit_transform(data)
    imputed_data = pd.DataFrame(data_imputed, columns=cols, index=idx)
    return imputed_data

def savgol_smoothing(data, window_length=10, polyorder=2):
    """
    Apply Savitzky-Golay filter for smoothing.
    Ensure window_length <= length of data, and polyorder < window_length.
    """
    smoothed = data.copy()
    for col in smoothed.columns:
        smoothed[col] = savgol_filter(smoothed[col], window_length, polyorder)
    return smoothed

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalization for each column, with protections against
    zero or NaN standard deviations by replacing them with 1.
    """
    data = data.copy()
    
    # Compute column-wise mean and std
    col_means = data.mean(axis=0)
    col_stds = data.std(axis=0)
    
    # Replace any zero or NaN std values with 1 to avoid division by zero
    col_stds.replace(0, 1, inplace=True)
    col_stds.fillna(1, inplace=True)

    # Z-score each column
    normalized = (data - col_means) / col_stds
    
    return normalized


def center_coordinate(data, reference_joint='KNE'):
    """
    Subtract the (X,Y) coords of a reference_joint in the first frame from all columns
    so that the reference_joint in the first frame is at (0,0).
    """
    centered = data.copy()
    ref_x = centered[f'{reference_joint}_X'].iloc[0]
    ref_y = centered[f'{reference_joint}_Y'].iloc[0]

    # Subtract the reference joint's X,Y from all X,Y columns
    x_cols = [col for col in centered.columns if '_X' in col]
    y_cols = [col for col in centered.columns if '_Y' in col]

    centered[x_cols] = centered[x_cols].values - ref_x
    centered[y_cols] = centered[y_cols].values - ref_y

    return centered

def scale_data(data):
    """
    Example scale using a 'KNE_ANK_length' as reference.
    Adjust to your domain or remove if not appropriate.
    """
    scaled = data.copy()
    if 'KNE_ANK_length' in scaled.columns:
        scale_factor = scaled['KNE_ANK_length'].mean()
        if scale_factor == 0:
            scale_factor = 1
    else:
        # fallback scale factor
        scale_factor = scaled.abs().max().max()
        if scale_factor == 0:
            scale_factor = 1

    for col in scaled.columns:
        if 'orientation' in col:
            col_min = scaled[col].min()
            col_max = scaled[col].max()
            if col_max != col_min:
                scaled[col] = (scaled[col] - col_min) / (col_max - col_min)
        else:
            scaled[col] = scaled[col] / scale_factor


    return scaled


# ------------------ Feature Computation ------------------ #
def compute_velocity(data, joints):
    """
    Compute X,Y velocity from position data. Also keep velocity magnitude if needed.
    Returns a DataFrame with columns:
      joint_Xvel, joint_Yvel, joint_VelMag, ...
    dropping edges (length/orientation).
    """
    # 1) Copy and take first derivative
    vel_data = data.copy().diff().fillna(0)


    # 3) For each joint, rename X-> xVel, Y-> yVel and keep magnitude
    new_cols = {}
    for joint in joints:
        x_col = f'{joint}_X'
        y_col = f'{joint}_Y'

        if x_col in vel_data.columns and y_col in vel_data.columns:
            xvel_name = f'{joint}_xVel'
            yvel_name = f'{joint}_yVel'
            vel_data[xvel_name] = vel_data[x_col]
            vel_data[yvel_name] = vel_data[y_col]
            # Magnitude
            vel_data[f'{joint}_VelMag'] = np.sqrt(vel_data[xvel_name]**2 + vel_data[yvel_name]**2)

            # We'll drop the old X,Y from this DF
            new_cols[x_col] = None
            new_cols[y_col] = None

    # Actually drop old X and Y
    vel_data.drop(columns=[c for c in new_cols if c in vel_data.columns], inplace=True)

    return vel_data

def compute_acceleration(vel_data, joints):
    """
    Compute acceleration from velocity DataFrame.
    For each joint, keep xAcc, yAcc, and AccMag.
    """
    acc_data = vel_data.copy().diff().fillna(0)

    # For each joint, rename xVel-> xAcc, yVel-> yAcc, then compute magnitude
    for joint in joints:
        xvel_col = f'{joint}_xVel'
        yvel_col = f'{joint}_yVel'
        velmag_col = f'{joint}_VelMag'
        
        if xvel_col in acc_data.columns and yvel_col in acc_data.columns:
            xacc_name = f'{joint}_xAcc'
            yacc_name = f'{joint}_yAcc'
            acc_data[xacc_name] = acc_data[xvel_col]
            acc_data[yacc_name] = acc_data[yvel_col]
            acc_data[f'{joint}_AccMag'] = np.sqrt(acc_data[xacc_name]**2 + acc_data[yacc_name]**2)

            # drop old velocity columns
            acc_data.drop(columns=[xvel_col, yvel_col], inplace=True, errors='ignore')

        # Also remove the old VelMag if it's there
        if velmag_col in acc_data.columns:
            acc_data.drop(columns=velmag_col, inplace=True, errors='ignore')

    return acc_data


# ------------------ Segmentation & Reshaping ------------------ #
def segment_data(data, label, window_size=100, overlap=0.5):
    """
    Segment data into overlapping windows of length `window_size` with given overlap.
    Ensures the final segment always includes the end of the data.

    data (DataFrame) : Input data with at least `window_size` rows.
    label (array-like): Label info used to construct segment label.
    window_size (int): Length of each segment window.
    overlap (float)  : Fraction of overlap between consecutive segments.

    Returns:
        segments (np.ndarray): shape = (num_segments, window_size, num_features)
        labels   (np.ndarray): shape = (num_segments, ?)  # expanded label info
    """
    num_samples = data.shape[0]
    stride = int(window_size * (1 - overlap))
    segments = []
    labels_list = []
    count = 0

    # If data is shorter than one window, return single segment
    if num_samples < window_size:
        segments.append(data.values)
        new_label = np.c_[
            label[None, :2],       # e.g. participant/leg IDs
            np.array([count]),     # segment index
            label[None, 2:]        # any other label columns
        ]
        labels_list.append(new_label)
        return np.array(segments), np.vstack(labels_list)

    last_start = 0
    for start in range(0, num_samples - window_size + 1, stride):
        last_start = start
        end = start + window_size
        segment = data.iloc[start:end]
        segments.append(segment.values)

        new_label = np.c_[
            label[None, :2],
            np.array([count]),
            label[None, 2:]
        ]
        labels_list.append(new_label)
        count += 1

    # If we didn't reach the end, add a final window
    if last_start + window_size < num_samples:
        start = num_samples - window_size
        end = num_samples
        segment = data.iloc[start:end]
        segments.append(segment.values)
        new_label = np.c_[
            label[None, :2],
            np.array([count]),
            label[None, 2:]
        ]
        labels_list.append(new_label)

    return np.array(segments), np.vstack(labels_list)

def reshape_data(array, X_cols, Y_cols, X_Vel_cols, X_Acc_cols, Y_Vel_cols, Y_Acc_cols):
    """
    Reorganize data into final shape: (batch, channels, frames, joints, 1-person).
    `array` shape is typically (batch, frames, features).
    You must slice out X, Y, Vel, Acc columns to build the 4 channels.
    """
    # array: (batch, frames, features)
    # We'll index along the last dimension (features) for each set of cols.
    X = array[..., X_cols]
    Y = array[..., Y_cols]
    X_Vel = array[..., X_Vel_cols]
    X_Acc = array[..., X_Acc_cols]
    Y_Vel = array[..., Y_Vel_cols]
    Y_Acc = array[..., Y_Acc_cols]

    # Stack into shape: (4, batch, frames, <something>), then add a dimension for M=1
    # and reorder to (batch, 4, frames, joints, 1)
    stacked = np.stack([X, Y, X_Vel, Y_Vel, X_Acc, Y_Acc], axis=1)  # => (batch, 6, frames, n_joints)
    input_data = stacked[..., None]               # => (batch, 6, frames, n_joints, 1)

    return input_data
