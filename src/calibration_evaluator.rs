//! Evaluating projector calibration quality by reprojecting the planar
//! calibration pattern and comparing against the camera-measured points.
//!
//! Given per-view extrinsics (rvec/tvec) and intrinsics, the known board
//! object points are reprojected and the residual against the measured points
//! is reported as mean/RMS reprojection error (in pixels). Results are written
//! out as per-view scatter plots (SVG + PNG) and a JSON dump per projector.

use std::{fs, path::Path};

use anyhow::{anyhow, Result};
use itertools::{multizip, Itertools};
use opencv::{
    calib3d,
    core::{Mat, Point2f, Point3f, Size, Vector},
    prelude::*,
};
use plotters::{
    prelude::*,
    style::text_anchor::{HPos, Pos, VPos},
};
use serde::{Deserialize, Serialize};
use tiny_skia::Pixmap;

/// One projector's calibration outputs to be evaluated.
///
/// `rvecs`/`tvecs` and `measured_points` are parallel per-view lists (one entry
/// per captured board pose).
#[derive(Debug, Clone)]
pub struct EvaluationData {
    pub projector_id: usize,
    pub camera_matrix: Mat,
    pub dist_coeffs: Mat,
    /// Per-view rotation vectors (Rodrigues).
    pub rvecs: Vec<Mat>,
    /// Per-view translation vectors.
    pub tvecs: Vec<Mat>,
    /// Per-view measured pattern points in image pixels.
    pub measured_points: Vec<Vec<Point2f>>,
    /// Caller-supplied metadata copied verbatim into the JSON output.
    pub additional_data_to_save: serde_json::Value,
}

/// A 2D measured point in image pixels, as serialized in the JSON dump.
#[derive(Serialize, Deserialize)]
struct MeasuredPoint {
    x: f32,
    y: f32,
}

/// On-disk JSON representation of one projector's evaluation (input/output).
///
/// Mats are flattened to nested `Vec`s so they round-trip through serde; the
/// error fields are computed on save only and skipped when loading.
#[derive(Serialize, Deserialize)]
struct EvaluationDataJson {
    camera_matrix: Vec<Vec<f64>>,
    dist_coeff: Vec<Vec<f64>>,
    rvecs: Vec<Vec<Vec<f64>>>,
    tvecs: Vec<Vec<Vec<f64>>>,
    measured_points: Vec<Vec<MeasuredPoint>>,
    /// Mean reprojection error in pixels (filled in on save, not loaded).
    #[serde(skip_deserializing)]
    mean_error: f64,
    /// RMS reprojection error in pixels (filled in on save, not loaded).
    #[serde(skip_deserializing)]
    rms_error: f64,
    #[serde(skip_deserializing)]
    additional_data: serde_json::Value,
}

/// Build the planar checkerboard object points (z = 0) in board coordinates.
///
/// `square_size` is the spacing between corners (mm), so returned coordinates
/// are in millimetres, ordered row-major over the `pattern_size` grid.
pub fn generate_object_points(pattern_size: Size, square_size: f32) -> Vector<Point3f> {
    let mut points = Vector::new();
    for i in 0..pattern_size.height {
        for j in 0..pattern_size.width {
            points.push(Point3f::new(
                j as f32 * square_size,
                i as f32 * square_size,
                0.0,
            ));
        }
    }
    points
}

/// Compute per-point reprojection residuals between measured and projected points.
///
/// Returns `(per_point_errors, mean_error, rms_error, max_abs_dx, max_abs_dy)`,
/// all in pixels. The two inputs are assumed equal length and aligned by index.
fn calculate_errors(
    measured: &Vector<Point2f>,
    projected: &Vector<Point2f>,
) -> (Vec<f32>, f32, f32, f32, f32) {
    let mut errors = Vec::new();
    let mut max_x_diff = 0.0_f32;
    let mut max_y_diff = 0.0_f32;

    for i in 0..measured.len() {
        let dx = measured.get(i).unwrap().x - projected.get(i).unwrap().x;
        let dy = measured.get(i).unwrap().y - projected.get(i).unwrap().y;
        let error = (dx * dx + dy * dy).sqrt();
        errors.push(error);

        max_x_diff = max_x_diff.max(dx.abs());
        max_y_diff = max_y_diff.max(dy.abs());
    }

    let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
    let rms_error = (errors.iter().map(|e| e * e).sum::<f32>() / errors.len() as f32).sqrt();

    (errors, mean_error, rms_error, max_x_diff, max_y_diff)
}

/// Render one view's projected-vs-measured scatter plot and write it as both SVG and PNG.
///
/// Files are written to `{output_path}/projector{projector_id}-{image_id}.{svg,png}`;
/// the PNG is rasterized from the SVG at 2x scale.
#[allow(clippy::too_many_arguments)]
fn plot_results(
    projected: &Vector<Point2f>,
    measured: &Vector<Point2f>,
    errors: &[f32],
    mean_error: f32,
    rms_error: f32,
    projector_id: usize,
    image_id: usize,
    output_path: &str,
) -> Result<()> {
    let svg_path = format!("{}/projector{}-{}.svg", output_path, projector_id, image_id);
    let png_path = format!("{}/projector{}-{}.png", output_path, projector_id, image_id);

    // Render the chart to SVG first
    draw_chart(
        &svg_path,
        projected,
        measured,
        errors,
        mean_error,
        rms_error,
        projector_id,
        image_id,
    )?;

    // Rasterize that SVG to PNG via usvg/resvg/tiny-skia
    let svg_data = fs::read(svg_path)?;

    // Parse the SVG (system fonts needed to resolve text labels)
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();

    let tree = usvg::Tree::from_data(&svg_data, &opt)?;

    // Render at 2x the SVG's intrinsic size for a crisper PNG
    let size = tree.size();
    let scale = 2.;
    let width = (size.width() * scale) as u32;
    let height = (size.height() * scale) as u32;

    let mut pixmap = Pixmap::new(width, height).ok_or(anyhow!("failed to create pixmap"))?;

    let transform = tiny_skia::Transform::from_scale(scale, scale);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // Save as PNG
    pixmap.save_png(png_path)?;

    Ok(())
}

/// Set up an SVG drawing area and delegate to `draw_chart_impl`.
///
/// Canvas is 740x480 px: 640 for the plot plus 100 for the error color bar.
#[allow(clippy::too_many_arguments)]
fn draw_chart(
    path: &str,
    projected: &Vector<Point2f>,
    measured: &Vector<Point2f>,
    errors: &[f32],
    mean_error: f32,
    rms_error: f32,
    projector_id: usize,
    image_id: usize,
) -> Result<()> {
    let root = SVGBackend::new(path, (740, 480)).into_drawing_area(); // +100px for the color bar
    draw_chart_impl(
        root,
        projected,
        measured,
        errors,
        mean_error,
        rms_error,
        projector_id,
        image_id,
    )
}

/// Draw the scatter plot, error-coded markers, legend, stats text, and color bar.
///
/// Markers: blue circles = projected, red crosses = measured; a Viridis-colored
/// circle at each pair's midpoint encodes the residual magnitude (0..2 px).
/// Y axis is inverted (`max_y..min_y`) to match image pixel coordinates.
#[allow(clippy::too_many_arguments)]
fn draw_chart_impl<DB: DrawingBackend>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    projected: &Vector<Point2f>,
    measured: &Vector<Point2f>,
    errors: &[f32],
    mean_error: f32,
    rms_error: f32,
    projector_id: usize,
    image_id: usize,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;

    // Split into the color-bar area and the plot area
    let (left, right) = root.split_horizontally(640);

    // Calculate data bounds
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for i in 0..projected.len() {
        let proj = projected.get(i).unwrap();
        let meas = measured.get(i).unwrap();

        min_x = min_x.min(proj.x).min(meas.x);
        max_x = max_x.max(proj.x).max(meas.x);
        min_y = min_y.min(proj.y).min(meas.y);
        max_y = max_y.max(proj.y).max(meas.y);
    }

    // Add margin (10% of range on each side)
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let x_margin = x_range * 0.1;
    let y_margin = y_range * 0.1;

    min_x -= x_margin;
    max_x += x_margin * 3.; // extra room on the right for the error stats text
    min_y -= y_margin;
    max_y += y_margin;

    // Draw the main chart
    let mut chart = ChartBuilder::on(&left)
        .caption(
            format!(
                "Projected vs Measured (Projector {} / State {})",
                projector_id, image_id
            ),
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, max_y..min_y)?;

    chart
        .configure_mesh()
        .x_desc("x [px]")
        .y_desc("y [px]")
        .draw()?;

    // Errors at/above this many pixels saturate the color scale
    let max_error = 2.0;

    // Draw an error-colored circle at each projected/measured midpoint
    for (proj, meas, &error) in multizip((projected.iter(), measured.iter(), errors.iter())) {
        let mx = (proj.x + meas.x) / 2.0;
        let my = (proj.y + meas.y) / 2.0;

        // Normalize residual to [0,1] for the Viridis lookup
        let t = (error / max_error).clamp(0.0, 1.0);
        let color = ViridisRGB::get_color(t);

        chart.draw_series(std::iter::once(Circle::new(
            (mx, my),
            6,
            color.mix(0.6).filled(),
        )))?;
    }

    // Draw projected points
    chart
        .draw_series(
            projected
                .iter()
                .map(|p| Circle::new((p.x, p.y), 3, BLUE.filled())),
        )?
        .label("Projected")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    // Draw measured points
    chart
        .draw_series(
            measured
                .iter()
                .map(|p| Cross::new((p.x, p.y), 3, RED.filled())),
        )?
        .label("Measured")
        .legend(|(x, y)| Cross::new((x, y), 3, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Draw error text
    let text_x = max_x;
    let text_y_start = min_y + y_margin * 0.5;

    chart.draw_series(std::iter::once(Text::new(
        format!("Mean Error: {:.2} px", mean_error),
        (text_x, text_y_start),
        ("sans-serif", 15)
            .into_font()
            .color(&BLACK)
            .pos(Pos::new(HPos::Right, VPos::Top)),
    )))?;

    chart.draw_series(std::iter::once(Text::new(
        format!("RMS Error: {:.2} px", rms_error),
        (text_x, text_y_start + y_range * 0.05),
        ("sans-serif", 15)
            .into_font()
            .color(&BLACK)
            .pos(Pos::new(HPos::Right, VPos::Top)),
    )))?;

    // Draw the color bar (top aligned with the body)
    let colorbar_steps = 100;
    let mut chart_colorbar = ChartBuilder::on(&right)
        .margin_top(50) // match the margin for the title
        .margin_bottom(50) // match the bottom too
        .margin_left(10)
        .margin_right(10)
        .set_label_area_size(LabelAreaPosition::Right, 40)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(colorbar_steps as f32))?;

    chart_colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .y_desc("Error [px]")
        .y_label_formatter(&|y| format!("{:.1}", (y / colorbar_steps as f32) * max_error))
        .label_style(("sans-serif", 14).into_font())
        .draw()?;

    // Draw the colormap (overlapping to fill gaps)
    chart_colorbar.draw_series((0..colorbar_steps).map(|i| {
        let t = i as f32 / colorbar_steps as f32;
        let color = ViridisRGB::get_color(t);

        // Overlap the rectangles slightly (i+1.5 fills the gaps)
        Rectangle::new(
            [
                (0.0, ((i + 1) as f32 + 0.5).min((i + 1) as f32)),
                (1.0, (i as f32 - 0.5).max(0.)),
            ],
            blend_with_white(color, 0.6).filled(),
        )
    }))?;

    // Draw the border
    let plotting_area_colorbar = chart_colorbar.plotting_area();
    plotting_area_colorbar.draw(&Rectangle::new(
        [(0.0, colorbar_steps as f32), (1.0, 0.0)],
        ShapeStyle::from(&BLACK),
    ))?;

    root.present()?;

    Ok(())
}

/// Alpha-composite `color` over a white background and return the opaque result.
fn blend_with_white(color: RGBColor, alpha: f64) -> RGBColor {
    // alpha: 0.0 (fully transparent) to 1.0 (opaque)
    let r = (alpha * (color.0 as f64) + (1.0 - alpha) * 255.0).round() as u8;
    let g = (alpha * (color.1 as f64) + (1.0 - alpha) * 255.0).round() as u8;
    let b = (alpha * (color.2 as f64) + (1.0 - alpha) * 255.0).round() as u8;
    RGBColor(r, g, b)
}

/// Evaluate each projector's calibration, writing per-view plots and a JSON dump.
///
/// For every view, reprojects the board object points and computes the residual
/// against the measured points; per-view plots and one `data{n}.json` per
/// projector are written under `output_dir` (created if needed). The aggregate
/// mean/RMS/max errors are printed to stdout.
pub fn save_camera_params_evaluation(
    camera_params: &[EvaluationData],
    pattern_size: Size,
    square_size: f32,
    output_dir: &Path,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let path = output_dir
        .to_str()
        .ok_or_else(|| anyhow!("output dir path is not valid UTF-8"))?;

    let object_points = generate_object_points(pattern_size, square_size);
    let mut total_measured_points = Vec::<Vec<Point2f>>::new();
    let mut total_projected_points = Vec::<Vec<Point2f>>::new();

    for (param_id, item) in camera_params.iter().enumerate() {
        println!("# Projector {}", item.projector_id);

        for (i, (rvec, tvec, measured_points)) in
            multizip((&item.rvecs, &item.tvecs, &item.measured_points)).enumerate()
        {
            let measured_points = measured_points.iter().copied().collect::<Vector<Point2f>>();
            let mut image_points = Vector::<Point2f>::new();
            calib3d::project_points(
                &object_points,
                rvec,
                tvec,
                &item.camera_matrix,
                &item.dist_coeffs,
                &mut image_points,
                &mut Mat::default(),
                0.0,
            )?;

            // Per-view residuals (max diffs are aggregated below, not per view)
            let (errors, mean_error, rms_error, _max_x_diff, _max_y_diff) =
                calculate_errors(&measured_points, &image_points);

            plot_results(
                &image_points,
                &measured_points,
                &errors,
                mean_error,
                rms_error,
                item.projector_id,
                i,
                path,
            )?;

            total_measured_points.push(measured_points.iter().collect_vec());
            total_projected_points.push(image_points.iter().collect_vec());
        }

        // Aggregate residuals over all views of this projector for the summary
        let (_errors, mean_error, rms_error, max_x_diff, max_y_diff) = calculate_errors(
            &total_measured_points
                .iter()
                .flatten()
                .copied()
                .collect::<Vector<Point2f>>(),
            &total_projected_points
                .iter()
                .flatten()
                .copied()
                .collect::<Vector<Point2f>>(),
        );
        println!("Max diff: [{:.2}, {:.2}]", max_x_diff, max_y_diff);
        println!("Mean reprojection error: {:.2} px", mean_error);
        println!("RMS reprojection error: {:.2} px", rms_error);
        let measured_points_vec: Vec<_> = total_measured_points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|x| MeasuredPoint { x: x.x, y: x.y })
                    .collect_vec()
            })
            .collect();

        let output_data = EvaluationDataJson {
            camera_matrix: item.camera_matrix.to_vec_2d::<f64>()?,
            dist_coeff: item.dist_coeffs.to_vec_2d::<f64>()?,
            rvecs: item
                .rvecs
                .iter()
                .map(|x| x.to_vec_2d::<f64>().unwrap())
                .collect_vec(),
            tvecs: item
                .tvecs
                .iter()
                .map(|x| x.to_vec_2d::<f64>().unwrap())
                .collect_vec(),
            measured_points: measured_points_vec,
            mean_error: mean_error as f64,
            rms_error: rms_error as f64,
            additional_data: item.additional_data_to_save.clone(),
        };

        let json_path = format!("{}/data{}.json", path, param_id);
        let json_str = serde_json::to_string_pretty(&output_data)?;
        fs::write(json_path, json_str)?;
    }

    Ok(())
}
