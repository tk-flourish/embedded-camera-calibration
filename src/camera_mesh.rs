//! Loading the optical-center compensation mesh produced by the
//! `prepare_camera_mesh` binary.

use std::{collections::HashMap, fs::File, path::Path};

use anyhow::Result;
use opencv::{
    calib3d::{find_homography, RANSAC},
    core::{Mat, Point2f, Vector},
};

use crate::types::{CameraId, CameraMeshPointPair};

/// Load the compensation mesh and build, per embedded camera, the homography
/// mapping camera-image coordinates to world (calibration-board) coordinates.
/// This realizes the paper's M_n (camera pixel c_n -> board point x_n),
/// estimated with RANSAC least-squares.
pub fn load_camera_mesh_homographies<P: AsRef<Path>>(path: P) -> Result<HashMap<CameraId, Mat>> {
    let mesh: HashMap<CameraId, Vec<CameraMeshPointPair>> =
        serde_json::from_reader(File::open(path)?)?;

    let mut homographies = HashMap::new();
    for (camera_id, pairs) in mesh {
        let world: Vector<Point2f> = pairs
            .iter()
            .map(|x| x.effective_camera_position_in_world)
            .collect();
        let camera: Vector<Point2f> = pairs
            .iter()
            .map(|x| x.projector_position_in_embedded_camera.to::<f32>().unwrap())
            .collect();
        let mut mask = Mat::default();
        let homography = find_homography(&camera, &world, &mut mask, RANSAC, 3.0)?;
        homographies.insert(camera_id, homography);
    }
    Ok(homographies)
}
