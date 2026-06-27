//! Build the optical-center compensation mesh (paper §4.1).
//!
//! The calibration board stays fixed while a single projector is moved to many
//! positions. At each projector position the operator triggers a capture that
//! pairs the projector point seen in an embedded camera with that camera's
//! effective position on the board (world mm), accumulating the per-camera
//! mesh. The result is written to `.data/camera_mesh.json` (default `--out`),
//! which is the prerequisite input for the other binaries in this crate.

use std::{collections::HashMap, fs, path::Path};

use anyhow::Result;
use clap::Parser;
use embedded_camera_calibration::{
    calibrator::{Calibrator, InitCameraData},
    debug_viz::{generate_point_pairs_debug, make_grid_image, show_debug},
    external_calibrator::ExternalCalibrator,
    types::CameraMeshPointPair,
};
use opencv::core::{Point2f, Point2i, Size};

/// CLI arguments.
#[derive(Parser)]
#[command(about = "Build the optical-center compensation mesh")]
struct Args {
    /// Output path for the compensation mesh.
    #[arg(long, default_value = ".data/camera_mesh.json")]
    out: String,

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

/// Entry point: accumulate per-projector-position captures into the compensation
/// mesh until the user quits.
fn main() -> Result<()> {
    // ---- parse CLI args & prepare output directory ----
    let args = Args::parse();
    if let Some(parent) = Path::new(&args.out).parent() {
        fs::create_dir_all(parent)?;
    }

    let projector_size = Size::new(1280, 800);
    let embedded_camera_size = Size::new(1920, 1080);
    let pattern_size = Size::new(args.checker_cols, args.checker_rows);
    let ext_calibrator = ExternalCalibrator::new(projector_size, pattern_size, args.square_size_mm)?;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct ProjectorImageResult {
        projector_position_in_embedded_camera: Point2i,
        embedded_camera_position_in_projector: Point2f,
    }

    // ---- per-projector-position capture loop (board fixed, projector moved) ----
    let mut point_pairs_per_projector: HashMap<usize, Vec<CameraMeshPointPair>> = HashMap::new();
    loop {
        // ---- connect to embedded cameras ----
        let mut calibrator = Calibrator::new(
            projector_size.width as u32,
            projector_size.height as u32,
            1,
            args.cameras
                .iter()
                .map(|address| InitCameraData {
                    address: address.clone(),
                })
                .collect::<Vec<_>>(),
        )?;

        // ---- projector-image side ----
        // Detect each embedded camera's projector-pixel position for this projector position.
        let Result::Ok(proj_img_res) = (|| -> Result<_> {
            let res = &calibrator.calibrate(500, 300, 500, 800)?[0];
            let res_subpix = &calibrator.calibrate_subpix(500, 300, 500)?[0];
            let res_final = res
                .cameras
                .iter()
                .filter_map(|(k, v)| {
                    res_subpix.get(k).map(|&v_subpix| {
                        (
                            *k,
                            ProjectorImageResult {
                                projector_position_in_embedded_camera: v
                                    .projector_position_in_camera,
                                embedded_camera_position_in_projector: v_subpix,
                            },
                        )
                    })
                })
                .collect::<HashMap<_, _>>();
            Ok(res_final)
        })() else {
            eprintln!("failed to get projector image coords.");
            continue;
        };

        // ---- world side ----
        // Recover each camera's effective position on the board (world mm).
        let Ok(world_res) = ext_calibrator.find_effective_camera_positions(
            &proj_img_res
                .iter()
                .map(|(&k, v)| (k, v.embedded_camera_position_in_projector))
                .collect::<HashMap<_, _>>(),
        ) else {
            eprintln!("failed to get projector world coords.");
            continue;
        };

        // ---- accumulate point pairs ----
        for (k, v_w) in world_res {
            let v_p = proj_img_res[&k].projector_position_in_embedded_camera;
            point_pairs_per_projector
                .entry(k)
                .or_default()
                .push(CameraMeshPointPair {
                    projector_position_in_embedded_camera: v_p,
                    effective_camera_position_in_world: v_w,
                });
        }

        // ---- save mesh ----
        // Persist after every capture so progress is never lost.
        fs::write(
            &args.out,
            serde_json::to_string(&point_pairs_per_projector)?,
        )?;
        eprintln!(
            "Saved {} cameras to {}",
            point_pairs_per_projector.len(),
            args.out
        );

        // ---- render debug grid & check for quit ----
        let mut debug_images = HashMap::new();
        for (&k, v) in point_pairs_per_projector.iter() {
            debug_images.insert(
                k,
                generate_point_pairs_debug(v, embedded_camera_size, &format!("Camera {k}"))?,
            );
        }
        let debug_image = make_grid_image(&debug_images, embedded_camera_size)?;
        let key = show_debug(&debug_image)?;

        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
