//! Debug visualization helpers (overlays, Hough lines, grid composition, saving).

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use chrono::Local;
use opencv::{
    core::*,
    highgui::*,
    imgcodecs::imwrite,
    imgproc::{self, *},
};

use crate::types::CameraMeshPointPair;

/// Draw Hough lines (rho/theta form) onto `image`, each in a cycling color.
/// Lines are extended far beyond the image bounds so they span the whole frame.
pub fn draw_debug_hough(image: &mut Mat, lines: &Vector<Vec3f>) -> Result<()> {
    for (idx, line_vec) in lines.iter().enumerate() {
        let rho = line_vec[0];
        let theta = line_vec[1];

        let a = theta.cos();
        let b = theta.sin();
        let x0 = a * rho;
        let y0 = b * rho;
        let scale = 100000.0; // make the endpoints long enough
        let pt1 = Point::new((x0 + scale * -b) as i32, (y0 + scale * a) as i32);
        let pt2 = Point::new((x0 - scale * -b) as i32, (y0 - scale * a) as i32);

        let color = get_opencv_color(idx);

        line(
            image, pt1, pt2, color, 2, // line thickness
            LINE_8, 0,
        )?;
    }
    Ok(())
}

/// Pick a distinct BGR color for index `idx`, cycling through a fixed palette.
fn get_opencv_color(idx: usize) -> Scalar {
    let colors = [
        Scalar::new(0.0, 0.0, 255.0, 0.0),   // red (BGR)
        Scalar::new(0.0, 255.0, 0.0, 0.0),   // green
        Scalar::new(255.0, 0.0, 0.0, 0.0),   // blue
        Scalar::new(0.0, 255.0, 255.0, 0.0), // yellow
        Scalar::new(255.0, 0.0, 255.0, 0.0), // magenta
        Scalar::new(255.0, 255.0, 0.0, 0.0), // cyan
    ];
    colors[idx % colors.len()]
}

/// Build a fresh per-run output directory path `{base}/{timestamp}`.
///
/// The directory itself is not created (callers do that, e.g. via [`save_debug`]).
/// Binaries use this so each run writes its artifacts into its own directory
/// (e.g. `.res/evaluate_homography/20251015-173542`).
pub fn timestamped_output_dir(base: &str) -> PathBuf {
    PathBuf::from(base).join(Local::now().format("%Y%m%d-%H%M%S").to_string())
}

/// Save `image` into `dir` (created if needed) with a timestamped filename.
pub fn save_debug(dir: &Path, image: &Mat) -> Result<()> {
    fs::create_dir_all(dir)?;
    // Sub-second precision so rapid captures within one run never collide.
    let formatted = Local::now().format("%Y%m%d-%H%M%S%.6f").to_string();
    imwrite(
        dir.join(format!("{formatted}.png")).to_str().unwrap(),
        image,
        &Vector::default(),
    )?;
    Ok(())
}

/// Show a downscaled copy of `image` in a window and block for a keypress.
/// Returns the key code from `wait_key`. (Use [`save_debug`] to persist frames.)
pub fn show_debug(image: &Mat) -> Result<i32> {
    let mut resized = Mat::default();
    resize(
        &image,
        &mut resized,
        Size::new(1280, 800),
        0.,
        0.,
        INTER_LINEAR,
    )?;
    imshow("Debug", &resized)?;
    let ret = wait_key(0)?;
    Ok(ret)
}

/// Render an embedded-camera-sized debug image marking each decoded point pair.
/// Each point is drawn as a white dot labelled with its world position (mm), with
/// `top_left_label` overlaid as a header.
pub fn generate_point_pairs_debug(
    points: &[CameraMeshPointPair],
    embedded_camera_size: Size,
    top_left_label: &str,
) -> opencv::Result<Mat> {
    // Black canvas.
    let mut canvas = Mat::zeros(
        embedded_camera_size.height,
        embedded_camera_size.width,
        CV_8UC3,
    )?
    .to_mat()?;

    for pair in points {
        let pt = pair.projector_position_in_embedded_camera;
        let label = format!(
            "({:.2}mm, {:.2}mm)",
            pair.effective_camera_position_in_world.x, pair.effective_camera_position_in_world.y
        );

        // White point.
        imgproc::circle(
            &mut canvas,
            Point::new(pt.x, pt.y),
            10,                                    // radius
            Scalar::new(255.0, 255.0, 255.0, 0.0), // white
            -1,                                    // filled
            imgproc::LINE_8,
            0,
        )?;

        // Label (below-right of the point).
        let text_org = Point::new(pt.x + 5, pt.y + 15);
        imgproc::put_text(
            &mut canvas,
            &label,
            text_org,
            imgproc::FONT_HERSHEY_SIMPLEX,
            1., // scale
            Scalar::new(255.0, 255.0, 255.0, 0.0),
            3, // thickness
            imgproc::LINE_AA,
            false,
        )?;
    }

    // Top-left label.
    let baseline = &mut 0;
    let text_size = imgproc::get_text_size(
        top_left_label,
        imgproc::FONT_HERSHEY_SIMPLEX,
        4.,
        3,
        baseline,
    )?;

    // Top-left position with margin (offset y by the text height).
    let org = Point::new(10, text_size.height + 10);
    imgproc::put_text(
        &mut canvas,
        top_left_label,
        org,
        imgproc::FONT_HERSHEY_SIMPLEX,
        4.,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        3,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(canvas)
}

/// Compose up to four debug images into a 2x2 grid with white borders.
/// Each entry is resized to `single_size`; missing ids leave a blank (border-colored) cell.
pub fn make_grid_image(debug_images: &HashMap<usize, Mat>, single_size: Size) -> Result<Mat> {
    let border_thickness = 3;
    let border_color = Scalar::new(255., 255., 255., 0.);
    let cols = 2;
    let rows = 2;

    // Canvas size including borders.
    let width = single_size.width * cols + border_thickness * (cols + 1);
    let height = single_size.height * rows + border_thickness * (rows + 1);
    let canvas = Mat::new_size_with_default(Size::new(width, height), CV_8UC3, border_color)?;

    // Place each cell.
    for row in 0..rows {
        for col in 0..cols {
            let id = row * cols + col;
            if let Some(src) = debug_images.get(&(id as usize)) {
                let mut resized = Mat::default();
                imgproc::resize(
                    src,
                    &mut resized,
                    single_size,
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;

                let x = border_thickness + col * (single_size.width + border_thickness);
                let y = border_thickness + row * (single_size.height + border_thickness);

                let roi = Rect::new(x, y, single_size.width, single_size.height);
                let mut roi_ref = Mat::roi(&canvas, roi)?.clone_pointee();
                resized.copy_to(&mut roi_ref)?;
            }
        }
    }

    Ok(canvas)
}
