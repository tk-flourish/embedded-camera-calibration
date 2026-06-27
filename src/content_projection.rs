//! Projecting content onto the calibration board through the estimated
//! projector homographies (shared by the homography-projection binaries).

use std::collections::HashMap;

use anyhow::Result;
use itertools::Itertools;
use opencv::{
    core::{
        gemm, perspective_transform, Mat, Point2f, Rect, Scalar, Size, Vector, BORDER_CONSTANT,
        BORDER_DEFAULT, CV_8UC3, DECOMP_LU,
    },
    imgproc::{self, get_perspective_transform, pyr_down, warp_perspective, INTER_AREA},
    prelude::*,
};

use crate::types::{CalibrationResult, CameraId};

/// Canvas layout (in millimetres) that content is placed onto before being
/// warped to each projector. Defaults to the prototype board; override via CLI.
#[derive(Clone, Copy, Debug)]
pub struct CanvasLayout {
    pub width_mm: i32,
    pub height_mm: i32,
    pub content_x_start_mm: i32,
    pub content_x_end_mm: i32,
    pub content_y_start_mm: i32,
}

impl Default for CanvasLayout {
    fn default() -> Self {
        Self {
            width_mm: 220,
            height_mm: 160,
            content_x_start_mm: 20,
            content_x_end_mm: 200,
            content_y_start_mm: 0,
        }
    }
}

impl CanvasLayout {
    /// Width of the content placement region (`x_end - x_start`), in mm.
    pub fn content_width_mm(&self) -> i32 {
        self.content_x_end_mm - self.content_x_start_mm
    }
}

/// Place `src` onto a board-sized canvas (in mm scaled by `pixel_per_mm`).
/// Scales `src` to the full content width preserving aspect ratio; if the
/// resulting height exceeds the content region it is cropped (not scaled) at the
/// bottom.
pub fn write_to_canvas(src: &Mat, layout: &CanvasLayout, pixel_per_mm: i32) -> Result<Mat> {
    let canvas_width_px = layout.width_mm * pixel_per_mm;
    let canvas_height_px = layout.height_mm * pixel_per_mm;
    let mut canvas = Mat::zeros(canvas_height_px, canvas_width_px, CV_8UC3)?.to_mat()?;

    let x_start_px = layout.content_x_start_mm * pixel_per_mm;
    let available_width_px = layout.content_width_mm() * pixel_per_mm;
    let y_start_px = layout.content_y_start_mm * pixel_per_mm;

    let aspect_ratio = src.cols() as f64 / src.rows() as f64;
    let target_width = available_width_px;
    let target_height = (target_width as f64 / aspect_ratio).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        src,
        &mut resized,
        Size::new(target_width, target_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let max_height = canvas_height_px - y_start_px;
    let final_height = target_height.min(max_height);

    let roi = Rect::new(x_start_px, y_start_px, target_width, final_height);
    let mut canvas_roi = Mat::roi_mut(&mut canvas, roi)?;
    let src_roi = Mat::roi(&resized, Rect::new(0, 0, target_width, final_height))?.clone_pointee();
    src_roi.copy_to(&mut canvas_roi)?;

    Ok(canvas)
}

/// High-quality `warpPerspective` with automatic pyramid downscaling.
///
/// When the projected area is much smaller than the source, the source is
/// progressively halved (`pyrDown`) before warping to reduce aliasing.
/// `margin` controls how aggressively to downscale (e.g. 2.0).
pub fn warp_with_pyramid_scale(
    src: &Mat,
    h: &Mat,
    dst_size: Size,
    margin: f64,
) -> opencv::Result<Mat> {
    let src_size = src.size()?;
    let (w, h_img) = (src_size.width as f64, src_size.height as f64);

    let src_pts = Vector::<Point2f>::from(vec![
        Point2f::new(0.0, 0.0),
        Point2f::new(w as f32, 0.0),
        Point2f::new(w as f32, h_img as f32),
        Point2f::new(0.0, h_img as f32),
    ]);

    // Project the source corners and measure the projected area (shoelace).
    let mut dst_pts = Vector::<Point2f>::new();
    perspective_transform(&src_pts, &mut dst_pts, h)?;
    let pts: Vec<Point2f> = dst_pts.iter().collect_vec();
    let mut area_proj = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area_proj += pts[i].x as f64 * pts[j].y as f64 - pts[i].y as f64 * pts[j].x as f64;
    }
    area_proj = 0.5 * area_proj.abs();

    // Decide the number of pyramid levels (powers of two).
    let area_src = w * h_img;
    let scale_target = ((area_proj * margin) / area_src).sqrt();
    let mut n_down = 0usize;
    let mut scale_total = 1.0;
    while scale_total * 0.5 > scale_target {
        scale_total *= 0.5;
        n_down += 1;
    }

    let mut scaled = src.clone();
    for _ in 0..n_down {
        let mut tmp = Mat::default();
        pyr_down(&scaled, &mut tmp, Size::default(), BORDER_DEFAULT)?;
        scaled = tmp;
    }

    // Compensate the homography for the downscale: H' = H * S^-1.
    let inv_scale = 1.0 / scale_total;
    let s_inv = Mat::from_slice_2d(&[
        [inv_scale, 0.0, 0.0],
        [0.0, inv_scale, 0.0],
        [0.0, 0.0, 1.0],
    ])?;
    let mut h_adj = Mat::default();
    gemm(h, &s_inv, 1.0, &Mat::default(), 0.0, &mut h_adj, 0)?;

    let mut dst = Mat::default();
    warp_perspective(
        &scaled,
        &mut dst,
        &h_adj,
        dst_size,
        INTER_AREA,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;

    Ok(dst)
}

/// Build, per projector, the homography mapping board (world, mm) coordinates to
/// projector pixels, using the optical-center compensation mesh.
///
/// For each projector: map each embedded camera's decoded projector position
/// into world coordinates via `camera_mesh_homographies` (the paper's M_n), then
/// fit a homography from those world points (scaled by `pixel_per_mm`) to the
/// subpixel projector coordinates. Note the direction is world -> projector (the
/// inverse of the paper's projector -> board map), since it is used to warp
/// board-space content into projector pixels.
pub fn projector_homographies_from_mesh(
    res: &CalibrationResult,
    res_subpix: &[HashMap<CameraId, Point2f>],
    camera_mesh_homographies: &HashMap<CameraId, Mat>,
    pixel_per_mm: i32,
) -> Result<Vec<Mat>> {
    let mut homographies = Vec::new();
    for (projector_id, entry) in res.iter().enumerate() {
        let mut world_positions = HashMap::new();
        for (&camera_id, camera_result) in entry.cameras.iter() {
            let mut world = Vector::<Point2f>::new();
            perspective_transform(
                &Vector::<Point2f>::from(vec![camera_result
                    .projector_position_in_camera
                    .to::<f32>()
                    .unwrap()]),
                &mut world,
                &camera_mesh_homographies[&camera_id],
            )?;
            world_positions.insert(camera_id, world.get(0)?);
        }

        let mut from = Vector::<Point2f>::new();
        let mut to = Vector::<Point2f>::new();
        for (&camera_id, point) in res_subpix[projector_id].iter() {
            from.push(world_positions[&camera_id] * pixel_per_mm as f32);
            to.push(point.to::<f32>().unwrap());
        }
        homographies.push(get_perspective_transform(&from, &to, DECOMP_LU)?);
    }
    Ok(homographies)
}
