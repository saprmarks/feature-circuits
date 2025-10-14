import os
import glob

# Remote directory
remote_dir = "/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits"

# List of file IDs
file_ids = [1052, 3631, 802, 3213, 389, 3588, 568, 641, 881, 3566, 2574, 1216, 1307, 648, 3908, 1179, 2920, 1428, 273, 1372, 1625, 2298, 2911, 644, 3693, 3112, 1544, 2289, 2449, 1166, 1225, 1603, 3316, 2267, 3477, 3184, 3748, 3806, 1872, 2800, 3064, 1762, 2722, 3459, 1414, 3820, 3940, 2352, 3873, 3601, 430, 2651, 1601, 2049, 1787, 1692, 1982, 3152, 2483, 2491, 455, 2266, 1814, 1959, 3162, 2470, 3823, 2751, 2236, 3592, 3060, 2308, 784, 1019, 3543, 3004, 3552, 926, 425, 1658, 3694, 878, 453, 1106, 3303, 1504, 821, 2689, 2823, 3130, 481, 511, 2953, 135, 3824, 3336, 2353, 1743, 1181, 3639]

# Function to check if a file exists
def file_exists(file_path):
    return os.path.isfile(file_path)

# Iterate over the file IDs and check if the corresponding files exist
missing_ids = []
for file_id in file_ids:
    pt_file = f"{remote_dir}/{file_id}_dict10_node0.1_edge0.01_n*_aggsum.pt"
    png_file = f"{remote_dir}/plots/{file_id}_dict10_node0.1_edge0.01_n*_aggsum.png"

    pt_exists = any(file_exists(f) for f in glob.glob(pt_file))
    png_exists = any(file_exists(f) for f in glob.glob(png_file))

    if not pt_exists or not png_exists:
        missing_ids.append(file_id)

# Print the missing IDs
if not missing_ids:
    print("All files exist for the provided IDs.")
else:
    print("The following IDs are missing corresponding files:")
    print(missing_ids)

