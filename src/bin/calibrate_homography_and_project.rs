//! Calibrate the projectors (via the embedded cameras and the compensation
//! mesh) for a single board pose, then project the requested content onto the
//! board through the estimated per-projector homographies.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use embedded_camera_calibration::{
    calibrator::{Calibrator, InitCameraData},
    camera_mesh::load_camera_mesh_homographies,
    content_projection::{
        projector_homographies_from_mesh, warp_with_pyramid_scale, write_to_canvas, CanvasLayout,
    },
    debug_viz::{save_debug, timestamped_output_dir},
    external_camera::default_external_camera,
    patterns::generate_checkered,
    projection::get_winname,
};
use opencv::{
    core::{Scalar, Size},
    highgui::{imshow, wait_key},
    imgcodecs,
};

#[derive(Parser)]
#[command(about = "Calibrate one pose and project content via homography")]
struct Args {
    /// Image file to project. If omitted, a generated checkerboard is used.
    #[arg(long)]
    content: Option<String>,

    /// Number of projectors.
    #[arg(long, default_value_t = 3)]
    projectors: usize,

    /// Embedded-camera server addresses (host:port), one per camera in camera-id
    /// order. Defaults to the prototype rig.
    #[arg(long, num_args = 4, default_values = [
        "192.168.0.101:58919",
        "192.168.0.102:58919",
        "192.168.0.103:58919",
        "192.168.0.104:58919",
    ])]
    cameras: Vec<String>,

    /// Pixels per millimetre for the content canvas.
    #[arg(long, default_value_t = 8)]
    pixel_per_mm: i32,

    /// Path to the optical-center compensation mesh.
    #[arg(long, default_value = ".data/camera_mesh.json")]
    mesh: String,

    /// Canvas width in millimetres.
    #[arg(long, default_value_t = 220)]
    canvas_width_mm: i32,

    /// Canvas height in millimetres.
    #[arg(long, default_value_t = 160)]
    canvas_height_mm: i32,

    /// Content region X start in millimetres.
    #[arg(long, default_value_t = 20)]
    content_x_start_mm: i32,

    /// Content region X end in millimetres.
    #[arg(long, default_value_t = 200)]
    content_x_end_mm: i32,

    /// Content region Y start in millimetres.
    #[arg(long, default_value_t = 0)]
    content_y_start_mm: i32,

    /// Square size (mm) of the generated fallback checkerboard (ignored when
    /// `--content` is given).
    #[arg(long, default_value_t = 10)]
    checker_square_mm: i32,

    /// After projecting, capture the result with the external camera and save it.
    #[arg(long)]
    capture: bool,

    /// Output directory for the `--capture` frame. Defaults to a fresh
    /// timestamped directory under `.res/calibrate_homography_and_project/`.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Distinct color per projector so that overlapping checkerboards are visible.
fn checker_color(projector_id: usize) -> Scalar {
    [
        Scalar::new(255., 0., 0., 0.),
        Scalar::new(0., 255., 0., 0.),
        Scalar::new(0., 0., 255., 0.),
    ][projector_id % 3]
}

/// Entry point: calibrate one pose, build homographies, project content, optionally capture.
fn main() -> Result<()> {
    // ---- parse CLI args ----
    let args = Args::parse();
    let projector_size = Size::new(1280, 800);
    let layout = CanvasLayout {
        width_mm: args.canvas_width_mm,
        height_mm: args.canvas_height_mm,
        content_x_start_mm: args.content_x_start_mm,
        content_x_end_mm: args.content_x_end_mm,
        content_y_start_mm: args.content_y_start_mm,
    };

    // ---- load camera mesh ----
    let camera_mesh_homographies = load_camera_mesh_homographies(&args.mesh)?;

    // ---- connect to embedded cameras & run calibration capture ----
    let mut calibrator = Calibrator::new(
        projector_size.width as u32,
        projector_size.height as u32,
        args.projectors,
        args.cameras
            .iter()
            .map(|address| InitCameraData {
                address: address.clone(),
            })
            .collect::<Vec<_>>(),
    )?;
    let res = calibrator.calibrate(500, 800, 1000, 1100)?;
    let res_subpix = calibrator.calibrate_subpix(500, 800, 1000)?;

    // ---- build per-projector homographies ----
    let homographies = projector_homographies_from_mesh(
        &res,
        &res_subpix,
        &camera_mesh_homographies,
        args.pixel_per_mm,
    )?;

    // ---- load content (or fall back to a generated checkerboard) ----
    let content = args
        .content
        .as_ref()
        .map(|path| imgcodecs::imread(path, imgcodecs::IMREAD_COLOR))
        .transpose()?;

    // ---- warp & project per projector ----
    for (projector_id, h) in homographies.iter().enumerate() {
        let image = match &content {
            Some(image) => image.clone(),
            None => generate_checkered(
                layout.height_mm * args.pixel_per_mm,
                layout.content_width_mm() * args.pixel_per_mm,
                checker_color(projector_id),
                args.checker_square_mm * args.pixel_per_mm,
            )?,
        };
        let canvas = write_to_canvas(&image, &layout, args.pixel_per_mm)?;
        let transformed = warp_with_pyramid_scale(&canvas, h, projector_size, 2.0)?;
        imshow(&get_winname(projector_id), &transformed)?;
        // `imshow` only paints on the next `wait_key`; pump the event loop so
        // each projector window actually updates as it is shown.
        wait_key(1)?;
    }

    // ---- optionally capture the projected result ----
    if args.capture {
        wait_key(300)?;
        let camera = default_external_camera()?;
        let out = args
            .out
            .unwrap_or_else(|| timestamped_output_dir(".res/calibrate_homography_and_project"));
        save_debug(&out, &camera.capture()?)?;
    } else {
        wait_key(0)?;
    }
    Ok(())
}
