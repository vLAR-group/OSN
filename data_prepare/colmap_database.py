import argparse
import os
import os.path as osp
import sqlite3
import pandas as pd
import numpy as np

from colmap_utils.database import COLMAPDatabase, blob_to_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--preproc_path", type=str)
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(0)

    data_root = args.data_root
    dataset_name = data_root.split('/')[-1]

    # Load original database
    preproc_path = osp.join(data_root, args.preproc_path)
    dbfile = osp.join(data_root, 'colmap_init', 'database.db')

    db_connection = sqlite3.connect(dbfile)

    # creating cursor
    cur = db_connection.cursor()

    # reading all table names
    table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    # get table list
    print(table_list)

    cameras_table = pd.read_sql("SELECT * FROM cameras", db_connection)
    cameras = {}
    for camera_id, model, width, height, params in cur.execute(
        "SELECT camera_id, model, width, height, params FROM cameras"):
        camera = {'model': model, 'width': width, 'height': height}
        camera['params'] = blob_to_array(params, np.float64)
        cameras[camera_id] = camera

    images_table = pd.read_sql("SELECT * FROM images", db_connection)
    n_view = len(images_table)
    images = dict(
        (image_id, image_name)
        for image_id, image_name in cur.execute(
            "SELECT image_id, name FROM images"))
    image_ids = list(images.keys())
    image_names = list(images.values())


    # Destinated new database
    trajectories = np.load(osp.join(preproc_path, 'trajectories.npy'))
    visibilities = np.load(osp.join(preproc_path, 'visibilities.npy'))
    traj_features = np.zeros((trajectories.shape[0], trajectories.shape[1], 128), dtype=np.uint8)
    segm_pred = np.load(osp.join(preproc_path, 'segm_preds.npy'))
    n_object = np.unique(segm_pred).shape[0]

    # Re-order objects according to sizes, so that 1-st object is likely to be BG
    obj_ids, obj_sizes = np.unique(segm_pred, return_counts=True)
    # obj_ids = obj_ids[np.argsort(obj_sizes)[::-1]]

    # Create a databse for each object
    for k, obj_id in enumerate(obj_ids):
        save_dbfile = osp.join(preproc_path, 'database%d.db' % (k))
        if osp.exists(save_dbfile):
            os.remove(save_dbfile)
        db = COLMAPDatabase.connect(save_dbfile)
        db.create_tables()

        # Create cameras table
        camera_id, camera = list(cameras.items())[0]    # Assume monocular camera
        db.add_camera(camera['model'], camera['width'], camera['height'], camera['params'], camera_id=camera_id)
        print(camera['params'])

        # Create images table
        image_ids = []
        for image_name in image_names:
            image_id = db.add_image(image_name, camera_id=camera_id)
            image_ids.append(image_id)

        # Convert trajectories to keypoints & matches
        traj, vis = trajectories[segm_pred == obj_id], visibilities[segm_pred == obj_id]
        keypoints, descriptors, matches = {}, {}, {}
        n_point = traj.shape[0]
        for i in range(n_point):
            kps, kp_vis = traj[i], vis[i]
            traj_feats = traj_features[i]
            kp_vis = np.where(kp_vis)[0]
            kps = kps[kp_vis]
            vis_image_ids = [image_ids[j] for j in kp_vis]

            # Add keypoints & descriptors
            for v, vid in enumerate(vis_image_ids):
                if vid not in keypoints:
                    keypoints[vid] = []
                keypoints[vid].append(kps[v])
                if vid not in descriptors:
                    descriptors[vid] = []
                descriptors[vid].append(traj_feats[v])

            # Add matches
            for v1, vid1 in enumerate(vis_image_ids[:-1]):
                for v2, vid2 in enumerate(vis_image_ids[(v1+1):]):
                    if (vid1, vid2) not in matches:
                        matches[(vid1, vid2)] = []
                    match_kps_id = [len(keypoints[vid1]) - 1, len(keypoints[vid2]) - 1]
                    matches[(vid1, vid2)].append(match_kps_id)

        # Create keypoints table
        for image_id, kps in keypoints.items():
            db.add_keypoints(image_id, np.array(kps, dtype=np.float32))

        # Create descriptors table
        for image_id, feats in descriptors.items():
            db.add_descriptors(image_id, np.array(feats, dtype=np.uint8))

        # Create matches table
        for (image_id1, image_id2), match_kps_ids in matches.items():
            db.add_matches(image_id1, image_id2, np.array(match_kps_ids, dtype=np.int32))

        # Save database to file
        db.commit()

        # # Load and check
        # db_connection = sqlite3.connect(save_dbfile)
        # cur = db_connection.cursor()
        # cameras_table = pd.read_sql("SELECT * FROM cameras", db_connection)
        # images_table = pd.read_sql("SELECT * FROM images", db_connection)
        # keypoints_table = pd.read_sql("SELECT * FROM keypoints", db_connection)
        # matches_table = pd.read_sql("SELECT * FROM matches", db_connection)