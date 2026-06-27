//! External-camera calibration acquisition and optical-center compensation.
//!
//! Using a separate (external) camera that observes the projected output and a
//! printed checkerboard, this module recovers where projector pixels land in the
//! board's world frame. Three coordinate spaces are in play: projector pixels
//! (px), external-camera image pixels (px), and the checkerboard world frame
//! (mm, with the grid spacing given by `millimetre_per_block`). Local
//! homographies fit per checkerboard cell bridge camera-image to world, while
//! Gray-code decoding bridges camera-image to projector pixels.

use std::{collections::HashMap, f64};

use anyhow::Result;
use itertools::Itertools;
use maplit::hashmap;
use opencv::{
    calib3d::{self, draw_chessboard_corners, find_homography, RANSAC},
    core::*,
    highgui::*,
    imgproc::{self, *},
    prelude::{
        GrayCodePatternTrait, GrayCodePatternTraitConst, GrayCodePattern_ParamsTrait,
        StructuredLightPatternTrait,
    },
    structured_light::{GrayCodePattern, GrayCodePattern_Params},
    ximgproc,
};
use ordered_float::NotNan;

use crate::{
    debug_viz::{draw_debug_hough, show_debug},
    external_camera::{
        default_external_camera, ApertureValue, ExternalCamera, IsoValue, ShutterSpeed,
    },
    patterns::{generate_axis_lines, Axis},
    projection::{get_winname, project_and_capture, setup_window},
};

/// Drives the external camera to acquire checkerboard- and projector-pixel
/// correspondences used for optical-center compensation.
pub struct ExternalCalibrator {
    camera: Box<dyn ExternalCamera>,
    /// Projector resolution in pixels.
    projector_size: Size,
    /// Checkerboard inner-corner grid size (cols x rows).
    pattern_size: Size,
    /// Physical size of one checkerboard square, in millimetres.
    millimetre_per_block: f32,
}

impl ExternalCalibrator {
    /// Construct a calibrator bound to the default external camera.
    pub fn new(
        projector_size: Size,
        pattern_size: Size,
        millimetre_per_block: f32,
    ) -> Result<ExternalCalibrator> {
        Ok(ExternalCalibrator {
            camera: default_external_camera()?,
            projector_size,
            pattern_size,
            millimetre_per_block,
        })
    }

    /// Capture the printed checkerboard and detect its corners in camera-image px.
    /// Returns the subpixel corners (top-left to bottom-right) plus a debug image;
    /// errors if no checkerboard is found.
    fn process_finding_checker_pattern(&self) -> Result<(Vector<Point2f>, Mat)> {
        // Project + capture (checkerboard pattern)
        // Channel separation
        // Checkerboard detection
        // Project a white frame first so the printed board is well lit for capture.
        self.camera.set_iso(IsoValue::Iso400)?;
        self.camera.set_aperture(ApertureValue::F160)?;
        self.camera.set_shutter_speed(ShutterSpeed::S60)?;

        let white = Mat::new_size_with_default(
            self.projector_size,
            CV_8UC3,
            Vec4d::new(255., 255., 255., 0.),
        )?;
        let black =
            Mat::new_size_with_default(self.projector_size, CV_8UC3, Vec4d::new(0., 0., 0., 0.))?;

        imshow(&get_winname(1), &white)?;
        wait_key(300)?;

        let cap = project_and_capture(self.camera.as_ref(), 0, &black)?;

        imshow(&get_winname(1), &black)?;
        wait_key(300)?;

        let mut red_channel = Mat::default();
        extract_channel(&cap, &mut red_channel, 2)?;

        let mut points_in_camera: Vector<Point2f> = Vector::new();
        let found = calib3d::find_chessboard_corners(
            &red_channel,
            self.pattern_size,
            &mut points_in_camera,
            calib3d::CALIB_CB_ADAPTIVE_THRESH,
        )?;
        if !found {
            anyhow::bail!("Checkerboard was not detected.");
        }

        corner_sub_pix(
            &red_channel,
            &mut points_in_camera,
            Size::new(5, 5),
            Size::new(-1, -1),
            TermCriteria::new(TermCriteria_EPS + TermCriteria_MAX_ITER, 30, 0.01)?,
        )?;
        // Normalize ordering so corners always run top-left -> bottom-right.
        if points_in_camera.iter().next().unwrap() > points_in_camera.iter().next_back().unwrap() {
            points_in_camera = points_in_camera
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .collect::<Vector<Point2f>>();
        }

        let mut debug = cap.clone();
        draw_chessboard_corners(&mut debug, self.pattern_size, &points_in_camera, found)?;

        Ok((points_in_camera, debug))
    }

    /// Intersect two lines given in Hough normal form (rho, theta), returning the
    /// crossing point in image px by solving the 2x2 linear system.
    fn intersection(rho1: f32, theta1: f32, rho2: f32, theta2: f32) -> opencv::Result<Point2f> {
        // Coefficient matrix: each row is the line normal [cos(theta), sin(theta)].
        let a = Mat::from_slice_2d(&[[theta1.cos(), theta1.sin()], [theta2.cos(), theta2.sin()]])?; // 2x2

        // Constant vector
        let b = Mat::from_slice_2d(&[[rho1], [rho2]])?; // 2x1

        let a_inv = a.inv(opencv::core::DECOMP_LU)?; // Inverse
        let x_mat = (&a_inv * &b).into_result()?.to_mat()?;

        let x = *x_mat.at_2d::<f32>(0, 0)?;
        let y = *x_mat.at_2d::<f32>(1, 0)?;

        Ok(Point2f::new(x, y))
    }

    /// Locate, in camera-image px, where a given projector pixel `position` lands.
    /// Projects a vertical and a horizontal line through that projector pixel,
    /// detects each in the capture, and returns their intersection.
    fn process_find_projector_pixel_on_camera_image(
        &self,
        position: Point2f,
        mut debug_image: Option<&mut Mat>,
    ) -> Result<Point2f> {
        //   Project + capture (X, Y)
        //   Line detection (X, Y)
        //   Find their intersection (image coordinates)

        // Reference rows/cols (projector px) for the two axis lines to project.
        let ref_positions = hashmap! {
            Axis::X => position.x.round() as i32,
            Axis::Y => position.y.round() as i32
        };
        let mut res = hashmap! {};
        self.camera.set_iso(IsoValue::Iso400)?;
        self.camera.set_aperture(ApertureValue::F160)?;
        self.camera.set_shutter_speed(ShutterSpeed::S30)?;

        for (axis, ref_position) in ref_positions {
            let pattern = generate_axis_lines(self.projector_size, &[ref_position], axis)?;
            let cap = project_and_capture(self.camera.as_ref(), 0, &pattern)?;

            let mut blue_channel = Mat::default();
            extract_channel(&cap, &mut blue_channel, 0)?;

            let mut binary = Mat::default();
            imgproc::threshold(
                &blue_channel,          // input image
                &mut binary,            // output image
                10.0,                   // threshold
                255.0,                  // max value
                imgproc::THRESH_BINARY, // threshold type
            )?;

            let kernel = imgproc::get_structuring_element(
                imgproc::MORPH_RECT,
                Size::new(3, 3),
                Point::new(-1, -1),
            )?;
            let mut dilated = Mat::default();
            dilate(
                &binary,
                &mut dilated,
                &kernel,
                Point::new(-1, -1),
                1,
                BORDER_CONSTANT,
                Scalar::default(),
            )?;

            // Threshold + dilate + thin to a one-pixel skeleton before Hough.
            let mut skeleton = Mat::default();
            ximgproc::thinning(&dilated, &mut skeleton, ximgproc::THINNING_ZHANGSUEN)?;

            let mut lines = Vector::<Vec3f>::new();
            hough_lines(
                &skeleton,
                &mut lines,
                1.,
                f64::consts::PI / 180.0 / 8.,
                500,
                0.,
                0.,
                0.,
                CV_PI,
                false,
            )?;

            if let Some(debug) = debug_image.as_mut() {
                draw_debug_hough(debug, &lines)?;
            }

            if lines.is_empty() {
                anyhow::bail!("Couldn't find any lines on {:?}.", axis);
            }

            // Keep the strongest line (highest accumulator vote, index 2).
            res.insert(
                axis,
                lines
                    .iter()
                    .sorted_by_key(|x| NotNan::new(-x.get(2).unwrap()).unwrap())
                    .next()
                    .unwrap(),
            );
        }
        // The two axis lines cross at the camera-image px of the projector pixel.
        let intersection = Self::intersection(
            res[&Axis::X][0],
            res[&Axis::X][1],
            res[&Axis::Y][0],
            res[&Axis::Y][1],
        )?;
        if let Some(debug) = debug_image {
            circle(
                debug,
                intersection.to::<i32>().unwrap(),
                24,
                Scalar::new(255., 255., 255., 0.),
                -1,
                LINE_8,
                0,
            )?;
        }

        Ok(intersection)
    }

    /// Find the checkerboard cell whose center is nearest the image point (px).
    /// Returns the cell's (row, col), i.e. the index of its top-left grid point.
    fn find_containing_cell(
        &self,
        point: Point2f,
        checker_points: &Vector<Point2f>,
    ) -> Result<(i32, i32)> {
        let mut min_dist = f32::MAX;
        let mut best_cell = (0, 0);
        let pattern_size = self.pattern_size;

        // Check each cell (one fewer than the grid points)
        for row in 0..(pattern_size.height - 1) {
            for col in 0..(pattern_size.width - 1) {
                // Get the four corner grid points of the cell
                let idx_tl = (row * pattern_size.width + col) as usize;
                let idx_tr = (row * pattern_size.width + col + 1) as usize;
                let idx_bl = ((row + 1) * pattern_size.width + col) as usize;
                let idx_br = ((row + 1) * pattern_size.width + col + 1) as usize;

                let tl = checker_points.get(idx_tl)?;
                let tr = checker_points.get(idx_tr)?;
                let bl = checker_points.get(idx_bl)?;
                let br = checker_points.get(idx_br)?;

                // Cell center
                let center_x = (tl.x + tr.x + bl.x + br.x) / 4.0;
                let center_y = (tl.y + tr.y + bl.y + br.y) / 4.0;

                let dx = point.x - center_x;
                let dy = point.y - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cell = (row, col);
                }
            }
        }

        Ok(best_cell)
    }

    /// Build camera-image px -> world mm correspondences for the 4x4 grid points
    /// around `center_cell` (clamped to the board), for fitting a local homography.
    fn get_local_correspondence(
        &self,
        center_cell: (i32, i32),
        checker_points: &Vector<Point2f>,
    ) -> Result<(Vector<Point2f>, Vector<Point2f>)> {
        let mut src_points = Vector::<Point2f>::new();
        let mut dst_points = Vector::<Point2f>::new();
        let pattern_size = self.pattern_size;

        let (cell_row, cell_col) = center_cell;

        // Get the 4x4 grid points surrounding the 3x3 cells
        // Take the 4x4 grid points from the cell top-left
        let start_row = (cell_row - 1).max(0);
        let start_col = (cell_col - 1).max(0);
        let end_row = (cell_row + 2).min(pattern_size.height - 1);
        let end_col = (cell_col + 2).min(pattern_size.width - 1);

        for row in start_row..=end_row {
            for col in start_col..=end_col {
                let idx = (row * pattern_size.width + col) as usize;

                // image coordinates
                let image_pt = checker_points.get(idx)?;
                src_points.push(image_pt);

                // world coordinates
                let world_x = col as f32 * self.millimetre_per_block;
                let world_y = row as f32 * self.millimetre_per_block;
                dst_points.push(Point2f::new(world_x, world_y));
            }
        }

        Ok((src_points, dst_points))
    }

    /// Map a camera-image px point to board world coordinates (mm).
    /// Fits a homography on the local checkerboard cell block: a piecewise-projective
    /// approximation that tracks local perspective / board non-planarity more
    /// accurately than one global homography would. (Lens distortion is not
    /// corrected here; the paper treats it as negligible over this small area.)
    fn image_to_world_coordinate(
        &self,
        image_point: Point2f,
        checker_points: &Vector<Point2f>,
        debug_image: Option<&mut Mat>,
    ) -> Result<Point2f> {
        // Find the grid cell containing the image point
        let cell_idx = self.find_containing_cell(image_point, checker_points)?;

        // Get the correspondences for a 3x3 cell block (4x4 grid points)
        let (src_points, dst_points) = self.get_local_correspondence(cell_idx, checker_points)?;

        // Compute the homography
        let homography =
            find_homography(&src_points, &dst_points, &mut Mat::default(), RANSAC, 3.0)?;

        // Transform image coordinates to world coordinates
        let cam_pt = [image_point].into_iter().collect::<Vector<Point2f>>();
        let mut proj_pt = Vector::default();
        perspective_transform(&cam_pt, &mut proj_pt, &homography)?;

        let decoded_proj: Point_<f32> = proj_pt.get(0).unwrap();

        if let Some(debug) = debug_image {
            put_text(
                debug,
                &format!("({:.02}mm, {:.02}mm)", decoded_proj.x, decoded_proj.y),
                image_point.to::<i32>().unwrap(),
                FONT_HERSHEY_SIMPLEX,
                2.,
                Scalar::new(0., 255., 0., 0.),
                4,
                LINE_8,
                false,
            )?;
        }

        Ok(decoded_proj)
    }

    /// Find each embedded camera's effective position on the board (the paper's
    /// x_n), in world mm, keyed by camera id.
    /// Captures the checkerboard once, then for each camera locates its projector
    /// pixel in the camera image and converts that to the world point where the
    /// ray meets the board; cameras that fail are skipped. Interactive: shows a
    /// debug image and bails if the user presses 'n'.
    pub fn find_effective_camera_positions(
        &self,
        cameras_in_projector_coord: &HashMap<usize, Point2f>,
    ) -> Result<HashMap<usize, Point2f>> {
        for i in 0..2 {
            setup_window(i, self.projector_size)?;
        }

        let (checker_points, mut debug_image) = self.process_finding_checker_pattern()?;

        let mut res = hashmap! {};
        for (&camera_id, position_in_projector) in cameras_in_projector_coord.iter() {
            let Ok(position_in_world) = (|| -> Result<_> {
                let position_in_camera = self.process_find_projector_pixel_on_camera_image(
                    *position_in_projector,
                    Some(&mut debug_image),
                )?;
                let position_in_world = self.image_to_world_coordinate(
                    position_in_camera,
                    &checker_points,
                    Some(&mut debug_image),
                )?;
                Ok(position_in_world)
            })() else {
                eprintln!("failed on camera {camera_id}");
                continue;
            };
            res.insert(camera_id, position_in_world);
        }

        let key = show_debug(&debug_image)?;
        if key == 'n' as i32 {
            anyhow::bail!("cancelled this input by user");
        }

        Ok(res)
    }
}

/// Detect, via an external camera, the projector-image coordinates that fall on
/// the printed checkerboard corners (the conventional-baseline acquisition).
///
/// Returns, per checkerboard corner, its decoded projector pixel (px). Detects
/// the board in the capture, decodes a Gray-code sequence to map camera px ->
/// projector px, and fits a local homography on a patch around each corner for a
/// subpixel result. Interactive: retries detection until a board is found or the
/// user presses 'q'.
pub fn get_projector_image_coordinate_on_checkerboard_corners(
    camera: &dyn ExternalCamera,
    projector_count: usize,
    target_projector: usize,
    projector_size: Size,
    pattern_size: Size,
) -> Result<Vector<Point2f>> {
    // Side length (px) of the per-corner patch used to fit the local homography.
    let patch_size = 64;
    // Blank every non-target projector so only the target lights the scene.
    let black = Mat::zeros(projector_size.height, projector_size.width, CV_8UC3)?.to_mat()?;
    for i in 0..projector_count {
        setup_window(i, projector_size)?;
        if i != target_projector {
            imshow(&get_winname(i), &black)?;
        }
    }
    wait_key(1)?;

    // Generate the Gray-code patterns (plus all-black/all-white shadow refs).
    let mut gray_params = GrayCodePattern_Params::default()?;
    gray_params.set_width(projector_size.width);
    gray_params.set_height(projector_size.height);
    let mut gray = GrayCodePattern::create(&gray_params)?;
    gray.set_white_threshold(5)?;
    let mut graycodes = Vector::<Mat>::new();
    gray.generate(&mut graycodes)?;
    let mut black_code = Mat::default();
    let mut white_code = Mat::default();
    gray.get_images_for_shadow_masks(&mut black_code, &mut white_code)?;

    // Detect the checkerboard in camera-image px, retrying on failure.
    let target_window = get_winname(target_projector);
    let (points_in_camera, image_size) = loop {
        imshow(&target_window, &black_code)?;
        camera.set_iso(IsoValue::Iso800)?;
        camera.set_aperture(ApertureValue::F40)?;
        camera.set_shutter_speed(ShutterSpeed::S60)?;

        let mut captured = project_and_capture(camera, target_projector, &black_code)?;
        let mut red_channel = Mat::default();
        extract_channel(&captured, &mut red_channel, 2)?;
        let mut points_in_camera: Vector<Point2f> = Vector::new();
        let found = calib3d::find_chessboard_corners(
            &red_channel,
            pattern_size,
            &mut points_in_camera,
            calib3d::CALIB_CB_ADAPTIVE_THRESH,
        )?;
        if found {
            imgproc::corner_sub_pix(
                &red_channel,
                &mut points_in_camera,
                Size::new(5, 5),
                Size::new(-1, -1),
                TermCriteria::new(TermCriteria_EPS + TermCriteria_MAX_ITER, 30, 0.01)?,
            )?;
            // Reorder from top-left to bottom-right.
            if points_in_camera.iter().next().unwrap()
                > points_in_camera.iter().next_back().unwrap()
            {
                points_in_camera = points_in_camera
                    .into_iter()
                    .collect_vec()
                    .into_iter()
                    .rev()
                    .collect::<Vector<Point2f>>();
            }
            calib3d::draw_chessboard_corners(
                &mut captured,
                pattern_size,
                &points_in_camera,
                found,
            )?;
            break (points_in_camera, captured.size()?);
        }

        eprintln!("No checkerboard was found. Press 'q' to cancel, or wait to retry ...");
        if wait_key(5000)? == 'q' as i32 {
            anyhow::bail!("checkerboard detection cancelled by user");
        }
    };

    camera.set_iso(IsoValue::Iso400)?;
    camera.set_aperture(ApertureValue::F160)?;
    camera.set_shutter_speed(ShutterSpeed::S30)?;

    // Project + capture one Gray-code frame, returning its blue channel.
    let capture_and_extract = |pattern: &Mat| -> Result<Mat> {
        let mut extracted = Mat::default();
        extract_channel(
            &project_and_capture(camera, target_projector, pattern)?,
            &mut extracted,
            0,
        )?;
        Ok(extracted)
    };

    let mut captured = Vector::<Mat>::new();
    for code in graycodes.iter() {
        let image = capture_and_extract(&code)?;
        captured.push(image);
    }

    // For each detected corner, decode the Gray code over a surrounding patch to
    // get camera-px -> projector-px pairs, fit a homography, and map the corner.
    let image_points_entry = points_in_camera
        .iter()
        .map(|point| -> Result<_> {
            let mut from = Vector::<Point2f>::new(); // camera-image px
            let mut to = Vector::<Point2f>::new(); // decoded projector px
            for x_offset in (-patch_size / 2)..=(patch_size / 2) {
                for y_offset in (-patch_size / 2)..=(patch_size / 2) {
                    let (x, y) = (
                        point.x.round() as i32 + x_offset,
                        point.y.round() as i32 + y_offset,
                    );
                    if !(0..image_size.width).contains(&x) || !(0..image_size.height).contains(&y) {
                        continue;
                    }
                    let mut proj_pixel = Point2i::default();
                    gray.get_proj_pixel(&captured, x, y, &mut proj_pixel)?;
                    from.push(Point2f::new(x as f32, y as f32));
                    to.push(proj_pixel.to::<f32>().unwrap());
                }
            }
            // Local camera-px -> projector-px homography, then map the exact corner.
            let mut mask = Mat::default();
            let h = find_homography(&from, &to, &mut mask, calib3d::RANSAC, 1.)?;
            let mut transformed = Vector::<Point2f>::new();
            perspective_transform(
                &[point].into_iter().collect::<Vector<Point2f>>(),
                &mut transformed,
                &h,
            )?;
            let res = transformed.get(0)?;
            Ok(res)
        })
        .collect::<Result<Vector<Point2f>>>();

    image_points_entry
}
