import numpy as np
import open3d as o3d


COLOR20 = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])


def build_colored_pointcloud(pc, color):
    """
    :param pc: (N, 3).
    :param color: (N, 3).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud

def build_pointcloud_segm(pc, segm):
    """
    :param pc: (N, 3).
    :param segm: (N,).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(COLOR20[segm % COLOR20.shape[0]] / 255.)
    return point_cloud


lines = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
box_colors = [[0, 1, 0] for _ in range(len(lines))]

def build_bbox3d(boxes):
    """
    :param boxes: List [(8, 3), ...].
    """
    line_sets = []
    for corner_box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(box_colors)
        line_sets.append(line_set)
    return line_sets

def bound_to_box(bounds):
    """
    :param bounds: List [(3, 2), ...].
    """
    boxes = []
    for bound in bounds:
        corner_box = np.array([[bound[0, 0], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 1]]])
        boxes.append(corner_box)
    return boxes


def build_segm_vis(segm, with_background=False):
    """
    :param segm: (H, W).
    """
    if with_background:
        colors = np.concatenate((COLOR20[-1:], COLOR20[:-1]), axis=0)
    else:
        colors = COLOR20
    colors = colors / 255.

    color_map = colors[segm % colors.shape[0]]
    return color_map


if __name__ == '__main__':
    meshes = []
    for i in range(20):
        color = COLOR20[i] / 255.
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        height = - (i // 5 * 3.0)
        width = i % 5 * 3.0
        mesh.translate([width, height, 0])
        mesh.paint_uniform_color(color)
        meshes.append(mesh)
    o3d.visualization.draw_geometries(meshes)