import open3d as o3d
import numpy as np
import argparse
import os
from matplotlib import cm
from numba import jit

def point_to_point_icp(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation

####################################################################################################
# Perform point to point icp with scaling for alignment and return 4x4 transformation


def point_to_point_icp_scaling(source: np.array, target: np.array, threshold: float, trans_init: np.array) -> np.array:
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-08, relative_rmse=1e-08, max_iteration=1000))
    return reg_p2l.transformation


####################################################################################################
# Clips the error to a specific range and maps the scaled error to a color
def colorize_by_error(error: np.array, error_range: list = [0, 1]) -> np.array:
    error = np.clip(error, a_min=error_range[0], a_max=error_range[1])
    mn, mx = error_range[0], error_range[1]
    error = (error - mn) / (mx - mn)
    return cm.turbo(error)[:, :3]

####################################################################################################
# Read 4x4 transformation matrix. Syntax: 1 0 0 0
#                                         0 1 0 0
#                                         0 0 1 0
#                                         0 0 0 1


def read_transformation_matrix(path_to_transformation_matrix: str) -> np.array:
    if not os.path.isfile(path_to_transformation_matrix):
        raise FileNotFoundError(
            f"Transformation matrix not found: {path_to_transformation_matrix}"
        )
    T = []
    with open(path_to_transformation_matrix, 'r') as f:
        for line in f:
            values = line.strip().split(" ")
            T.append(values)

    return np.array(T).astype(float)


@jit(nopython=True, parallel = True, cache=True)  
def np_all_axis0(x):
    """Numba compatible version of np.all(x, axis=0)."""
    out = np.ones(x.shape[1], dtype=np.bool_)
    for i in range(x.shape[0]):
        out = np.logical_and(out, x[i, :])
    return out
    
    
@jit(nopython=True)  
def crop_with_convex_polygon_numba(points, polygon):      
     
    edges = polygon[1:] - polygon[:-1]    
    verts_to_points = points[np.newaxis, :, :] - polygon[:-1, np.newaxis, :]
    
    cross_vals = edges[:, np.newaxis, 0] * verts_to_points[:, :, 1] \
                 - edges[:, np.newaxis, 1] * verts_to_points[:, :, 0]

    first_sign = np.sign(cross_vals[0])    

    same_side = np_all_axis0(np.sign(cross_vals) == first_sign)    
    
    return same_side

def crop_with_convex_polygon(pcd, polygon):    
    
    polygon_points = np.concatenate((polygon, np.ones((len(polygon), 1))* 1000), axis=-1)
    polygon_points = np.append(polygon_points, np.concatenate((polygon, np.ones((len(polygon), 1))* -1000), axis=-1),axis=0)  
    polygon_pcd = o3d.geometry.PointCloud()
    polygon_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    obb = polygon_pcd.get_oriented_bounding_box()    
    pcd = pcd.crop(obb)
    
    points = np.array(pcd.points)
    points = points[:,:2]
    
    same_side = np.zeros(len(points))
    
    step_size = 10_000_000
    
    offset = 0    
    while offset < len(points):  
        same_side[offset: offset + step_size] = crop_with_convex_polygon_numba(points[offset: offset + step_size], polygon)
        offset += step_size
    
    indices = np.where(same_side)[0]

    return pcd.select_by_index(indices)

def eval(source_pcd, target_pcd, completeness_threshold, abs_error_threshold, print_csv = False):
    
    distance1 = np.array(source_pcd.compute_point_cloud_distance(target_pcd))
    distance2 = np.array(target_pcd.compute_point_cloud_distance(source_pcd))
    
    recall = len(distance2[distance2<abs_error_threshold]) / len(distance2)
    
    precision = len(distance1[distance1<abs_error_threshold]) / len(distance1)

    precision_recall_sum = recall + precision
    if precision_recall_sum == 0:
        fscore = 0.0
    else:
        fscore = 2 * recall * precision / precision_recall_sum * 100
    
    l1 = distance1[distance1 <= np.quantile(distance1, completeness_threshold)]
    ch = distance1.mean() + distance2.mean()
    
    mse = (l1 ** 2)
    rmse = np.sqrt(mse.mean())  
    
    if print_csv:
        print(f"{recall},{l1.mean()},{mse.mean()},{rmse},{ch},{fscore},",end="")
        return distance1

    
    print("abs: ", l1.mean())
    print("mse: ", mse.mean())
    print("rmse: ", rmse)        
    print("completeness", recall)
    print("FScore", fscore)
    print("ChamferDistance", ch)
    
    return distance1
    


####################################################################################################
# Print text header
def printHeader(estPath: str, gtPath: str, completeness_threshold: float, abs_error_threshold: float, perform_icp_alignment: bool, visualize_abs_error: bool, visualize_destination_path: str, path_to_transformation_matrix: str, polygon_path: str, voxel_size: float) -> None:
    print(f'################################################################################')
    print(f'# > Estimate: {estPath}')
    print(f'# > Ground truth: {gtPath}')
    print(f'# > Completeness threshold: {completeness_threshold}')
    print(f'# > Absolute error threshold: {abs_error_threshold}')
    print(f'# > With ICP adjustment: {perform_icp_alignment}')
    if visualize_abs_error:
        print(f'# > Visualize error: {visualize_abs_error}')
        print(f'# > Destination path for the color coded point cloud: {visualize_destination_path}')
    if path_to_transformation_matrix:
        print(f'# > Path to transformation matrix: {path_to_transformation_matrix}')
    if polygon_path:
        print(f'# > Path to polygon: {polygon_path}')
    if voxel_size:
        print(f'# > Voxel downsampling with voxel size: {voxel_size}')   

    
    print(f'# -------------------------------------------------------------------------------')


####################################################################################################
# Main entry point into application
if __name__ == "__main__":

    # initialize argument parser
    argParser = argparse.ArgumentParser(
        description="Utility script to evaluate point clouds.")
    argParser.add_argument("-icp", "--perform_icp_alignment", action='store_true',
                           help="Option to specify wether to use icp for better alignment. "
                           "The point clouds must already be at least roughly aligned for this to work.")

    argParser.add_argument("-vis", "--visualize_abs_error", action='store_true',
                           help="Option to visualize the absolut error color coded. "
                           "Saves a color coded point cloud.")

    argParser.add_argument("-vis_dest", "--visualize_destination_path", type=str, default="vis.ply",
                           help="Destination path for the color coded point cloud.")

    argParser.add_argument("-transform", "--path_to_transformation_matrix", type=str, default=None,
                           help="Destination path for a 4x4 matrix to transform the estimated point cloud.")

    argParser.add_argument("-transform_out", "--output_path_to_transformation_matrix", type=str, default=None,
                           help="Output path for the 4x4 matrix estimated with ICP.")

    argParser.add_argument("-cpl", "--completeness_threshold", type=float, default=0.9,
                           help="Completeness threshold used for calculating the error metrics. "
                           "For the final metrics only the lowest X percent of values are considered. ")

    argParser.add_argument("-abs", "--abs_error_threshold", type=float, default=0.2,
                           help="Error Threshold used to calculate completeness. "
                           "All data points <  abs_error_threshold are treated as inliers.")
    
    argParser.add_argument("-p", "--polygon_for_crop", type=str, default=None,
                           help="Destination path for a polygon to crop gt and estimation point cloud.")
    
    argParser.add_argument("-csv", "--print_csv", action=argparse.BooleanOptionalAction)

    argParser.add_argument("-vox", "--voxel_size", type=float, default=None,)


    argParser.add_argument(
        "estimate", help="Path to the estimate(s) that is/are to be evaluated.")

    argParser.add_argument(
        "ground_truth", help="Path to the data used as ground.")
    args = argParser.parse_args()

    # get input paths
    estPath = args.estimate
    gtPath = args.ground_truth

    # get optional args
    completeness_threshold = args.completeness_threshold
    abs_error_threshold = args.abs_error_threshold
    perform_icp_alignment = args.perform_icp_alignment
    visualize_abs_error = args.visualize_abs_error
    visualize_destination_path = args.visualize_destination_path
    path_to_transformation_matrix = args.path_to_transformation_matrix
    output_path_to_transformation_matrix = args.output_path_to_transformation_matrix
    polygon_for_crop = args.polygon_for_crop
    
    if args.print_csv is None:
        printHeader(estPath, gtPath, completeness_threshold, abs_error_threshold, perform_icp_alignment,
                visualize_abs_error, visualize_destination_path, path_to_transformation_matrix, polygon_for_crop, args.voxel_size)

    target_pcd = o3d.io.read_point_cloud(gtPath)
    
    if polygon_for_crop:   
        polygon = np.loadtxt(polygon_for_crop, dtype=float)  
        target_pcd = crop_with_convex_polygon(target_pcd, polygon)        
    
    translate = -np.asarray(target_pcd.get_center())
    target_pcd = target_pcd.translate(translate)
    
    if len(target_pcd.points) == 0:
        raise ValueError(f"Ground truth point cloud is empty: {gtPath}")

    source_pcd = o3d.io.read_point_cloud(estPath)

    if len(source_pcd.points) == 0:
        raise ValueError(f"Estimated point cloud is empty: {estPath}")
 
    if path_to_transformation_matrix:
        source_pcd = source_pcd.translate(translate)
        T = read_transformation_matrix(path_to_transformation_matrix)
        source_pcd = source_pcd.transform(T)
        source_pcd = source_pcd.translate(-translate)
        
    if polygon_for_crop:
        polygon = np.loadtxt(polygon_for_crop, dtype=float)
        source_pcd = crop_with_convex_polygon(source_pcd, polygon)    

    source_pcd = source_pcd.translate(translate)
    number_of_points = len(source_pcd.points)
    
    if args.voxel_size is not None:
        target_pcd = target_pcd.voxel_down_sample(args.voxel_size)
        source_pcd = source_pcd.voxel_down_sample(args.voxel_size)

    if perform_icp_alignment:
        if args.print_csv is None:
            print("--------- Error without icp alignment and completeness threshold: ", completeness_threshold)    

        if args.print_csv is None:
            print("--------- Error after icp alignment and completeness threshold: ",
              completeness_threshold)
            distance1 = eval(source_pcd, target_pcd, completeness_threshold, abs_error_threshold)    
        else:    
            distance1 = np.array(source_pcd.compute_point_cloud_distance(target_pcd))           
        
        transform_icp = point_to_point_icp(
            source_pcd, target_pcd, distance1.mean(), np.eye(4))
        source_pcd = source_pcd.transform(transform_icp)

        output_transform = np.eye(4)
        if path_to_transformation_matrix:
            output_transform = read_transformation_matrix(path_to_transformation_matrix)

        output_transform = transform_icp @ output_transform

        if args.print_csv is None:
            print("--------- Error after icp alignment and completeness threshold: ",
              completeness_threshold)
            distance1 = eval(source_pcd, target_pcd, completeness_threshold, abs_error_threshold)    
        else:    
            distance1 = np.array(source_pcd.compute_point_cloud_distance(target_pcd))     

        transform_icp = point_to_point_icp_scaling(
            source_pcd, target_pcd, distance1.mean(), np.eye(4))
        source_pcd = source_pcd.transform(transform_icp)

        output_transform = transform_icp @ output_transform

        C = np.eye(4)
        C[:3, 3] = translate 

        output_transform_world = output_transform

        if output_path_to_transformation_matrix:
            np.savetxt(output_path_to_transformation_matrix, output_transform_world)

        if args.print_csv is None:
            print("--------- Error after icp alignment with scaling and completeness threshold: ",
              completeness_threshold)  
    
    distance1 = eval(
        source_pcd,
        target_pcd,
        completeness_threshold,
        abs_error_threshold,
        print_csv=args.print_csv is not None,
    )
    
    if args.print_csv is None:
        print("Number of points:", len(np.asarray(source_pcd.points)))
    else:
        print(number_of_points)        
        
    if visualize_abs_error:   
        colors = colorize_by_error(distance1, error_range=[0, abs_error_threshold * 10])
        source_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(visualize_destination_path, source_pcd.translate(-translate))

    

