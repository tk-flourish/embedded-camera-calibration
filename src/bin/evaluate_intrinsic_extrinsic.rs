//! Calibrate each projector's intrinsic/extrinsic parameters (paper §4.4).
//!
//! Collects world<->projector point correspondences over multiple board poses
//! for three methods (conventional external-camera, proposed mesh-based, and
//! proposed without optical-center fix), runs Zhang's calibration per method,
//! and saves the resulting parameters together with RMS reprojection error.
//!
//! Requires the compensation mesh `.data/camera_mesh.json`.

use std::{f32, io, io::Write, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use embedded_camera_calibration::{
    calibration_evaluator::{
        generate_object_points, save_camera_params_evaluation, EvaluationData,
    },
    calibrator::{Calibrator, InitCameraData},
    camera_mesh::load_camera_mesh_homographies,
    debug_viz::timestamped_output_dir,
    external_calibrator::get_projector_image_coordinate_on_checkerboard_corners,
    external_camera::default_external_camera,
    types::parse_board_point_mm,
};
use itertools::{multizip, Itertools};
use opencv::{
    calib3d::{
        calibrate_camera, CALIB_FIX_K1, CALIB_FIX_K2, CALIB_FIX_K3, CALIB_ZERO_TANGENT_DIST,
    },
    core::{
        perspective_transform, Mat, Point2f, Point3f, Scalar, Size, TermCriteria,
        TermCriteria_COUNT, TermCriteria_EPS, Vector,
    },
    highgui::wait_key,
    imgproc,
};
use serde_json::json;

/// CLI arguments.
#[derive(Parser)]
#[command(about = "Calibrate each projector's intrinsic/extrinsic parameters")]
struct Args {
    /// Output directory for the evaluation results. Each method is written to a
    /// `conventional/`, `proposed/`, or `proposed_unfixed/` subdirectory of it.
    /// Defaults to a fresh timestamped directory under
    /// `.res/evaluate_intrinsic_extrinsic/`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Embedded-camera positions on the board, in world millimetres, one `x,y`
    /// per camera in camera-id order. Rig-specific (used for the uncorrected
    /// baseline) — see the README for the prototype's values.
    #[arg(long, value_parser = parse_board_point_mm, num_args = 4, required = true, value_name = "X,Y")]
    camera_positions: Vec<Point2f>,

    /// Embedded-camera server addresses (host:port), one per camera in camera-id
    /// order. Defaults to the prototype rig.
    #[arg(long, num_args = 4, default_values = [
        "192.168.0.101:58919",
        "192.168.0.102:58919",
        "192.168.0.103:58919",
        "192.168.0.104:58919",
    ])]
    cameras: Vec<String>,

    /// Checkerboard inner corners along the width (columns).
    #[arg(long, default_value_t = 12)]
    checker_cols: i32,

    /// Checkerboard inner corners along the height (rows).
    #[arg(long, default_value_t = 9)]
    checker_rows: i32,

    /// Checkerboard square size in millimetres.
    #[arg(long, default_value_t = 20.0)]
    square_size_mm: f32,
}

/// World<->projector correspondences for a single board pose.
#[derive(Debug, Default, Clone)]
struct PointsPair {
    pub points_in_world: Vec<Point3f>,
    pub points_in_projector: Vec<Point2f>,
}

/// Accumulated per-pose correspondences for the three calibration methods.
#[derive(Debug, Default, Clone)]
struct ConventionalProposed {
    conventional: Vec<PointsPair>,
    proposed: Vec<PointsPair>,
    proposed_unfixed: Vec<PointsPair>,
}

impl ConventionalProposed {
    /// Number of poses collected (all three methods stay in lockstep).
    pub fn len(&self) -> usize {
        self.conventional.len()
    }

    /// Append one pose's correspondences for all three methods.
    pub fn push(
        &mut self,
        conventional: PointsPair,
        proposed: PointsPair,
        proposed_unfixed: PointsPair,
    ) {
        self.conventional.push(conventional);
        self.proposed.push(proposed);
        self.proposed_unfixed.push(proposed_unfixed);
    }
}

/// Entry point: collect correspondences over poses, calibrate per method, save results.
fn main() -> Result<()> {
    // ---- parse CLI args & resolve output directory ----
    let args = Args::parse();
    let out = args
        .out
        .unwrap_or_else(|| timestamped_output_dir(".res/evaluate_intrinsic_extrinsic"));
    let camera_positions = args.camera_positions;

    // ---- parameters ----
    let projector_size = Size::new(1280, 800);
    let projector_count = 3;
    let pattern_size = Size::new(args.checker_cols, args.checker_rows);
    let square_size = args.square_size_mm;

    // ---- load camera mesh ----
    let camera_image_to_world_homographies =
        load_camera_mesh_homographies(".data/camera_mesh.json")?;

    // ---- connect to embedded cameras ----
    let mut calibrator = Calibrator::new(
        projector_size.width as u32,
        projector_size.height as u32,
        projector_count,
        args.cameras
            .iter()
            .map(|address| InitCameraData {
                address: address.clone(),
            })
            .collect::<Vec<_>>(),
    )?;

    let mut proposed_projection_time = std::time::Duration::new(0, 0);
    let mut conventional_projection_time = std::time::Duration::new(0, 0);

    // ---- collect correspondences over board poses (until 8 per projector) ----
    let mut data = vec![ConventionalProposed::default(); projector_count];
    while data.iter().any(|x| x.len() < 8) {
        eprint!("# Next {{ {} }}: ", data.iter().map(|x| x.len()).join(", "));
        io::stderr().flush()?;
        wait_key(0)?;
        println!();

        // ---- run calibration capture (mesh-based, timed) ----
        let timer = Instant::now();
        let res = calibrator.calibrate(500, 500, 800, 800)?;
        let res_subpix = calibrator.calibrate_subpix(500, 500, 800)?;

        println!("{:?}", res_subpix.iter().map(|x| x.len()).collect_vec());
        proposed_projection_time += timer.elapsed();

        for (projector_id, (data, res, res_subpix)) in
            multizip((data.iter_mut(), res.iter(), res_subpix.iter())).enumerate()
        {
            println!("## Projector {projector_id}");
            let succeeded = res_subpix.len() >= 4;

            if !succeeded {
                continue;
            }

            // ---- proposed: map embedded-camera detections to world via mesh ----
            let proposed = {
                let mut world = Vec::<Point2f>::new();
                let mut projector = Vec::<Point2f>::new();
                for (id, subpix_value) in res_subpix.iter() {
                    let res_value = res.cameras[id];

                    let mut perspective_res = Vector::<Point2f>::new();
                    perspective_transform(
                        &[res_value.projector_position_in_camera.to::<f32>().unwrap()]
                            .into_iter()
                            .collect::<Vector<Point2f>>(),
                        &mut perspective_res,
                        &camera_image_to_world_homographies[id],
                    )?;

                    world.push(perspective_res.get(0).unwrap());
                    projector.push(subpix_value.to::<f32>().unwrap());
                }
                PointsPair {
                    points_in_world: world
                        .iter()
                        .map(|item| Point3f::new(item.x, item.y, 0.))
                        .collect_vec(),
                    points_in_projector: projector,
                }
            };

            // ---- proposed_unfixed: same, but fixed nominal camera positions ----
            let proposed_unfixed = {
                let mut world = Vec::<Point2f>::new();
                let mut projector = Vec::<Point2f>::new();

                for (id, subpix_value) in res_subpix.iter() {
                    world.push(camera_positions[*id]);
                    projector.push(subpix_value.to::<f32>().unwrap());
                }
                PointsPair {
                    points_in_world: world
                        .iter()
                        .map(|item| Point3f::new(item.x, item.y, 0.))
                        .collect_vec(),
                    points_in_projector: projector,
                }
            };

            // ---- conventional: detect corners with the external camera (timed) ----
            let conventional = loop {
                // Wait until the external camera becomes available.
                let external_camera = loop {
                    eprint!("Waiting for the external camera ('q' to cancel) ...");
                    io::stderr().flush()?;
                    let key = wait_key(1000)?;
                    eprintln!();
                    if let Ok(external_camera) = default_external_camera() {
                        break external_camera;
                    }
                    if key == 'q' as i32 {
                        anyhow::bail!("cancelled while waiting for the external camera");
                    }
                };
                let timer = Instant::now();
                let projector = get_projector_image_coordinate_on_checkerboard_corners(
                    external_camera.as_ref(),
                    projector_count,
                    projector_id,
                    projector_size,
                    pattern_size,
                )?;
                let elapsed = timer.elapsed();
                // Project the debug image
                let mut img = Mat::new_rows_cols_with_default(
                    projector_size.height,
                    projector_size.width,
                    opencv::core::CV_8UC3,
                    Scalar::all(255.),
                )?;
                for point in projector.iter() {
                    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);
                    imgproc::circle(
                        &mut img,
                        point.to::<i32>().unwrap(),
                        2,
                        color,
                        -1,
                        imgproc::LINE_AA,
                        0,
                    )?;
                }
                calibrator.project_raw(&img, projector_id)?;
                if wait_key(0)? == 'n' as i32 {
                    continue;
                }
                conventional_projection_time += elapsed;
                let world = generate_object_points(pattern_size, square_size);
                break PointsPair {
                    points_in_projector: projector.into(),
                    points_in_world: world.into(),
                };
            };

            // ---- store this pose for all three methods ----
            data.push(conventional, proposed, proposed_unfixed);
        }
        println!(
            "proposed: {:.3}s, conventional: {:.3}s",
            proposed_projection_time.as_secs_f64(),
            conventional_projection_time.as_secs_f64()
        );
    }

    // ---- calibrate each projector per method (Zhang) ----
    let mut proposed_results = Vec::<EvaluationData>::new();
    let mut proposed_unfixed_results = Vec::<EvaluationData>::new();
    let mut conventional_results = Vec::<EvaluationData>::new();
    for (projector_id, data) in data.iter().enumerate() {
        for (data_entry, res, projection_time, fix_dist_coeffs_to_zero) in [
            (
                &data.conventional,
                &mut conventional_results,
                conventional_projection_time,
                false,
            ),
            (
                &data.proposed,
                &mut proposed_results,
                proposed_projection_time,
                true,
            ),
            (
                &data.proposed_unfixed,
                &mut proposed_unfixed_results,
                proposed_projection_time,
                true,
            ),
        ] {
            let mut camera_matrix = Mat::default();
            let mut dist_coeffs = Mat::default();
            let mut rvecs = Vector::<Mat>::default();
            let mut tvecs = Vector::<Mat>::default();
            calibrate_camera(
                &data_entry
                    .iter()
                    .map(|x| Vector::from_iter(x.points_in_world.iter().copied()))
                    .collect::<Vector<Vector<Point3f>>>(),
                &data_entry
                    .iter()
                    .map(|x| Vector::from_iter(x.points_in_projector.iter().copied()))
                    .collect::<Vector<Vector<Point2f>>>(),
                projector_size,
                &mut camera_matrix,
                &mut dist_coeffs,
                &mut rvecs,
                &mut tvecs,
                if fix_dist_coeffs_to_zero {
                    CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST
                } else {
                    0
                },
                TermCriteria::new(TermCriteria_COUNT + TermCriteria_EPS, 30, f64::EPSILON)?,
            )?;

            // Evaluate against THIS method's own measured projector points so the
            // per-view residuals/plots align with the rvecs/tvecs just fitted.
            let measured_points = data_entry
                .iter()
                .map(|x| x.points_in_projector.clone())
                .collect_vec();
            res.push(EvaluationData {
                projector_id,
                camera_matrix,
                dist_coeffs,
                rvecs: rvecs.into(),
                tvecs: tvecs.into(),
                measured_points,
                additional_data_to_save: json!({
                    "elapsed_secs": projection_time.as_secs_f64(),
                    "raw_data": data_entry.iter().map(|x| {
                        json!({
                            "points_in_projector": x.points_in_projector.iter().map(|y| (y.x, y.y)).collect_vec(),
                            "points_in_world": x.points_in_world.iter().map(|y| (y.x, y.y, y.z)).collect_vec()
                        })
                    }).collect_vec()
                }),
            });
        }
    }

    // ---- save results (params + RMS reprojection error) per method ----
    save_camera_params_evaluation(
        &conventional_results,
        pattern_size,
        square_size,
        &out.join("conventional"),
    )?;
    save_camera_params_evaluation(
        &proposed_results,
        pattern_size,
        square_size,
        &out.join("proposed"),
    )?;
    save_camera_params_evaluation(
        &proposed_unfixed_results,
        pattern_size,
        square_size,
        &out.join("proposed_unfixed"),
    )?;

    Ok(())
}
