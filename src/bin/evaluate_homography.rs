//! Two-projector alignment evaluation via homography (paper §4.3.1).
//!
//! Compares three calibration conditions (conventional, proposed, proposed
//! without optical-center compensation) by projecting overlapping red/green
//! checkerboards plus USAF-1951 and wave patterns for MTF measurement, and
//! capturing each with the external camera. Also captures each projector in
//! isolation for reference.
//!
//! Requires the compensation mesh `.data/camera_mesh.json` and the resolution
//! targets under `.data/` (`USAF-1951.png`, `waves/`).

use std::{
    borrow::Cow,
    io::{stdout, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::Parser;
use embedded_camera_calibration::{
    calibrator::{Calibrator, InitCameraData},
    camera_mesh::load_camera_mesh_homographies,
    content_projection::{
        projector_homographies_from_mesh, warp_with_pyramid_scale, write_to_canvas, CanvasLayout,
    },
    debug_viz::{save_debug, timestamped_output_dir},
    external_calibrator::get_projector_image_coordinate_on_checkerboard_corners,
    external_camera::{default_external_camera, ApertureValue, IsoValue, ShutterSpeed},
    patterns::generate_checkered,
    projection::get_winname,
    types::parse_board_point_mm,
};
use itertools::Itertools;
use opencv::{
    calib3d::find_homography,
    core::{Mat, Point2f, Scalar, Size, Vector, CV_8UC3, DECOMP_LU},
    highgui::{imshow, wait_key},
    imgcodecs,
    imgproc::get_perspective_transform,
};
use strum::{EnumIter, IntoEnumIterator};

/// CLI arguments.
#[derive(Parser)]
#[command(about = "Two-projector alignment evaluation via homography")]
struct Args {
    /// Output directory for the captured frames. Defaults to a fresh
    /// timestamped directory under `.res/evaluate_homography/`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Embedded-camera positions on the board, in world millimetres, one `x,y`
    /// per camera in camera-id order. Rig-specific (used for the uncorrected
    /// condition) — see the README for the prototype's values.
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
}

/// Entry point: build homographies per condition, project test patterns, and capture them.
fn main() -> Result<()> {
    // ---- parse CLI args & resolve output directory ----
    let args = Args::parse();
    let out = args
        .out
        .unwrap_or_else(|| timestamped_output_dir(".res/evaluate_homography"));
    let camera_positions = args.camera_positions;
    let layout = CanvasLayout {
        width_mm: args.canvas_width_mm,
        height_mm: args.canvas_height_mm,
        content_x_start_mm: args.content_x_start_mm,
        content_x_end_mm: args.content_x_end_mm,
        content_y_start_mm: args.content_y_start_mm,
    };

    // ---- parameters & constant canvases ----
    let projector_size = Size::new(1280, 800);
    let projector_count = 2;
    let pattern_size = Size::new(args.checker_cols, args.checker_rows);
    let square_size = args.square_size_mm;
    let pixel_per_millimetre = 8;
    // Projected checkerboard squares are half the printed board's squares.
    let projected_checker_stride = (square_size / 2.0 * pixel_per_millimetre as f32) as i32;
    let black = Mat::new_size_with_default(projector_size, CV_8UC3, Scalar::new(0., 0., 0., 0.))?;
    let _white =
        Mat::new_size_with_default(projector_size, CV_8UC3, Scalar::new(255., 255., 255., 0.))?;

    // ---- build conventional (external-camera) homographies ----
    let mut conventional_homography = Vector::<Mat>::new();
    for projector_id in 0..projector_count {
        let from = Vector::<Point2f>::from(
            (0..pattern_size.height)
                .flat_map(|y| {
                    (0..pattern_size.width).map(move |x| {
                        Point2f::new(
                            x as f32 * square_size * pixel_per_millimetre as f32,
                            y as f32 * square_size * pixel_per_millimetre as f32,
                        )
                    })
                })
                .collect_vec(),
        );
        let to = get_projector_image_coordinate_on_checkerboard_corners(
            default_external_camera()?.as_ref(),
            projector_count,
            projector_id,
            projector_size,
            pattern_size,
        )?;
        let mut mask = Mat::default();
        let transform_mat = find_homography(&from, &to, &mut mask, 0, 3.0)?;

        conventional_homography.push(transform_mat);
    }

    // ---- load camera mesh ----
    let camera_image_to_world_homographies =
        load_camera_mesh_homographies(".data/camera_mesh.json")?;

    // ---- connect to embedded cameras & run calibration capture ----
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
    let res = calibrator.calibrate(500, 800, 1000, 1100)?;
    let res_subpix = calibrator.calibrate_subpix(500, 800, 1000)?;

    // The three calibration conditions being compared.
    #[derive(Debug, EnumIter, PartialEq, Eq)]
    enum Condition {
        Conventional,
        Proposed,
        ProposedUncorrected,
    }

    // ---- wait for the operator to place the projection target ----
    print!("Place paper ...");
    stdout().flush()?;
    wait_key(0)?;
    println!();

    // ---- per-condition: build homographies, project patterns, capture ----
    for condition in Condition::iter() {
        // Select/compute the per-projector homographies for this condition.
        let homography_list = match condition {
            Condition::Conventional => Cow::Borrowed(&conventional_homography),
            Condition::Proposed => Cow::Owned(
                projector_homographies_from_mesh(
                    &res,
                    &res_subpix,
                    &camera_image_to_world_homographies,
                    pixel_per_millimetre,
                )?
                .into_iter()
                .collect::<Vector<Mat>>(),
            ),
            Condition::ProposedUncorrected => {
                // Fixed nominal camera positions (no optical-center compensation).
                let mut homographies = Vector::<Mat>::new();
                for entry in res_subpix.iter().take(projector_count) {
                    let mut from = Vector::<Point2f>::new();
                    let mut to = Vector::<Point2f>::new();
                    for (&camera_id, point) in entry.iter() {
                        from.push(camera_positions[camera_id] * pixel_per_millimetre as f32);
                        to.push(point.to::<f32>().unwrap());
                    }
                    homographies.push(get_perspective_transform(&from, &to, DECOMP_LU)?);
                }
                Cow::Owned(homographies)
            }
        };
        println!("Condition: {condition:?}");
        let external_camera = default_external_camera()?;

        // ---- capture black reference (bright exposure) ----
        external_camera.set_iso(IsoValue::Iso400)?;
        external_camera.set_aperture(ApertureValue::F40)?;
        external_camera.set_shutter_speed(ShutterSpeed::S60)?;
        for (projector_id, h) in homography_list.iter().enumerate() {
            let image = write_to_canvas(&black, &layout, pixel_per_millimetre)?;
            let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
            imshow(&get_winname(projector_id), &transformed)?;
        }
        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // ---- capture overlapping red/green checkerboards (alignment) ----
        imshow(
            &get_winname(0),
            &warp_with_pyramid_scale(
                &write_to_canvas(
                    &generate_checkered(
                        layout.height_mm * pixel_per_millimetre,
                        layout.content_width_mm() * pixel_per_millimetre,
                        Scalar::new(0., 0., 255., 0.),
                        projected_checker_stride,
                    )?,
                    &layout,
                    pixel_per_millimetre,
                )?,
                &homography_list.get(0).unwrap(),
                projector_size,
                2.0,
            )?,
        )?;
        imshow(
            &get_winname(1),
            &warp_with_pyramid_scale(
                &write_to_canvas(
                    &generate_checkered(
                        layout.height_mm * pixel_per_millimetre,
                        layout.content_width_mm() * pixel_per_millimetre,
                        Scalar::new(0., 255., 0., 0.),
                        projected_checker_stride,
                    )?,
                    &layout,
                    pixel_per_millimetre,
                )?,
                &homography_list.get(1).unwrap(),
                projector_size,
                2.0,
            )?,
        )?;
        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // ---- capture USAF-1951 & wave targets (MTF, sharp exposure) ----
        external_camera.set_iso(IsoValue::Iso400)?;
        external_camera.set_aperture(ApertureValue::F160)?;
        external_camera.set_shutter_speed(ShutterSpeed::S30)?;

        let usaf1951 = imgcodecs::imread(".data/USAF-1951.png", imgcodecs::IMREAD_COLOR)?;
        for (projector_id, h) in homography_list.iter().enumerate() {
            let image = write_to_canvas(&usaf1951, &layout, pixel_per_millimetre)?;
            let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
            imshow(&get_winname(projector_id), &transformed)?;
        }
        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // Sweep the vertical and horizontal sine-wave gratings for MTF.
        for entry in glob::glob(".data/waves/vertical/*.png")?
            .chain(glob::glob(".data/waves/horizontal/*.png")?)
        {
            let path = entry?;
            let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
            for (projector_id, h) in homography_list.iter().enumerate() {
                let image = write_to_canvas(&img, &layout, pixel_per_millimetre)?;
                let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
                imshow(&get_winname(projector_id), &transformed)?;
            }
            wait_key(300)?;
            let capture = external_camera.capture()?;
            save_debug(&out, &capture)?;
        }
    }

    // ---- per-projector single-projector reference captures ----
    for projector_id in 0..projector_count {
        println!("Condition: Single {projector_id}");
        // Blank every other projector so only this one contributes.
        for i in 0..projector_count {
            if i == projector_id {
                continue;
            }
            imshow(&get_winname(i), &black)?;
        }
        let external_camera = default_external_camera()?;
        let h = conventional_homography.get(projector_id).unwrap();

        // ---- capture black reference (bright exposure) ----
        external_camera.set_iso(IsoValue::Iso400)?;
        external_camera.set_aperture(ApertureValue::F40)?;
        external_camera.set_shutter_speed(ShutterSpeed::S60)?;

        let image = write_to_canvas(&black, &layout, pixel_per_millimetre)?;
        let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
        imshow(&get_winname(projector_id), &transformed)?;
        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // ---- capture white checkerboard ----
        imshow(
            &get_winname(projector_id),
            &warp_with_pyramid_scale(
                &write_to_canvas(
                    &generate_checkered(
                        layout.height_mm * pixel_per_millimetre,
                        layout.content_width_mm() * pixel_per_millimetre,
                        Scalar::new(255., 255., 255., 0.),
                        projected_checker_stride,
                    )?,
                    &layout,
                    pixel_per_millimetre,
                )?,
                &h,
                projector_size,
                2.0,
            )?,
        )?;

        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // ---- capture USAF-1951 & wave targets (MTF, sharp exposure) ----
        external_camera.set_iso(IsoValue::Iso400)?;
        external_camera.set_aperture(ApertureValue::F160)?;
        external_camera.set_shutter_speed(ShutterSpeed::S30)?;

        let usaf1951 = imgcodecs::imread(".data/USAF-1951.png", imgcodecs::IMREAD_COLOR)?;
        let image = write_to_canvas(&usaf1951, &layout, pixel_per_millimetre)?;
        let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
        imshow(&get_winname(projector_id), &transformed)?;
        wait_key(300)?;
        let capture = external_camera.capture()?;
        save_debug(&out, &capture)?;

        // Sweep the vertical and horizontal sine-wave gratings for MTF.
        for entry in glob::glob(".data/waves/vertical/*.png")?
            .chain(glob::glob(".data/waves/horizontal/*.png")?)
        {
            let path = entry?;
            let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
            let image = write_to_canvas(&img, &layout, pixel_per_millimetre)?;
            let transformed = warp_with_pyramid_scale(&image, &h, projector_size, 2.0)?;
            imshow(&get_winname(projector_id), &transformed)?;
            wait_key(300)?;
            let capture = external_camera.capture()?;
            save_debug(&out, &capture)?;
        }
    }

    Ok(())
}
