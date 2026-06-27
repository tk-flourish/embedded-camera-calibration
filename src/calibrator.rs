use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    net::ToSocketAddrs,
    ops::RangeInclusive,
    time::Instant,
};

use anyhow::{anyhow, Result};
use itertools::Itertools;
use opencv::{core::*, highgui::*, prelude::*, structured_light::*};
use rayon::prelude::*;

use crate::{
    camera::*,
    patterns::{generate_axis_lines, Axis},
    projection::{get_winname, setup_window},
    types::*,
};

pub struct Camera {
    connection: CameraConnection,
}

pub struct InitCameraData<A>
where
    A: ToSocketAddrs + Send + Sync + Clone,
{
    pub address: A,
}

pub struct Calibrator {
    width: u32,
    height: u32,
    graycodes: Vector<Mat>,
    black_code: Mat,
    white_code: Mat,
    projector_count: usize,
    cameras: Vec<Camera>,
    start_time: Option<Instant>,
    client_timestamps: VecDeque<u32>,
    result: Option<CalibrationResult>,
}

impl Calibrator {
    /// Initialize
    pub fn new<A>(
        width: u32,
        height: u32,
        projector_count: usize,
        cameras: Vec<InitCameraData<A>>,
    ) -> Result<Self>
    where
        A: ToSocketAddrs + Send + Sync + Clone,
    {
        let mut params = GrayCodePattern_Params::default()?;
        params.set_width(width as i32);
        params.set_height(height as i32);

        let mut pattern = GrayCodePattern::create(&params)?;
        pattern.set_white_threshold(5)?;

        let mut graycodes = Vector::<Mat>::new();
        pattern.generate(&mut graycodes)?;

        let mut black_code = Mat::default();
        let mut white_code = Mat::default();
        pattern.get_images_for_shadow_masks(&mut black_code, &mut white_code)?;

        let mut instance = Calibrator {
            width,
            height,
            graycodes,
            black_code,
            white_code,
            projector_count,
            cameras: vec![],
            start_time: None,
            client_timestamps: VecDeque::new(),
            result: None,
        };

        instance.register_cameras(cameras)?;

        Ok(instance)
    }

    /// Bit width of the projector ID code
    fn projector_id_bit_width(&self) -> u32 {
        1.max(usize::BITS - (self.projector_count - 1).leading_zeros())
    }

    /// Number of Gray-code patterns along the X axis
    fn graycode_x_count(&self) -> u32 {
        u32::BITS - self.width.leading_zeros()
    }

    /// Number of Gray-code patterns along the Y axis
    fn graycode_y_count(&self) -> u32 {
        u32::BITS - self.height.leading_zeros()
    }

    /// Block (via `wait_key`) until `time` ms have elapsed since `start`.
    /// Used to keep projection in step with the cameras' scheduled timestamps.
    fn wait_until(time: u32, start: Instant) -> Result<()> {
        wait_key(time as i32 - start.elapsed().as_millis() as i32)?;
        Ok(())
    }

    /// Build the capture schedule (ms timestamps) for one calibration run:
    /// a black/white pair, an area-calculation gap, then two frames per
    /// Gray-code bit (projector id + X + Y). `for_client` adds the host's
    /// trailing readout frame.
    fn generate_timestamps(
        projector_id_bit_width: u32,
        graycode_x_count: u32,
        graycode_y_count: u32,
        start_delay: u32,
        interval: u32,
        area_calculation_interval: u32,
        for_client: bool,
    ) -> Vec<u32> {
        let mut data = vec![];
        let mut time = start_delay;

        // black white
        for _ in 0..2 {
            data.push(time);
            time += interval;
        }
        time += area_calculation_interval - interval;
        for _ in 0..(2 * (projector_id_bit_width + graycode_x_count + graycode_y_count)
            + if for_client { 1 } else { 0 })
        {
            data.push(time);
            time += interval;
        }

        data
    }

    /// Register cameras
    fn register_cameras<A>(&mut self, cameras: Vec<InitCameraData<A>>) -> Result<()>
    where
        A: ToSocketAddrs + Send + Sync + Clone,
    {
        for camera in cameras {
            let connection = CameraConnection::new(camera.address.clone())?;
            self.cameras.push(Camera { connection });
        }
        Ok(())
    }

    /// Measure each camera's RTT, then compute and distribute the capture
    /// timestamp schedule so all cameras sample the patterns in lock-step.
    /// The host keeps its own copy (`client_timestamps`) to time projection.
    fn set_timestamps(
        &mut self,
        start_delay: u32,
        capture_delay: u32,
        interval: u32,
        area_calculation_interval: u32,
    ) -> Result<()> {
        let rtt_vec = {
            let mut rtt_vec = vec![Option::<u32>::None; self.cameras.len()];
            self.cameras
                .par_iter_mut()
                .zip(rtt_vec.par_iter_mut())
                .for_each(|(camera, rtt)| {
                    *rtt = camera.connection.rtt(10).ok();
                });
            if rtt_vec.iter().any(|item| item.is_none()) {
                return Err(anyhow!("Failed to get all RTT"));
            }
            rtt_vec.iter().map(|item| item.unwrap()).collect::<Vec<_>>()
        };

        let projector_id_bit_width = self.projector_id_bit_width();
        let graycode_x_count = self.graycode_x_count();
        let graycode_y_count = self.graycode_y_count();

        let max_rtt = *rtt_vec.iter().max().unwrap();

        self.client_timestamps = VecDeque::from_iter(Self::generate_timestamps(
            projector_id_bit_width,
            graycode_x_count,
            graycode_y_count,
            max_rtt + start_delay,
            interval,
            area_calculation_interval,
            true,
        ));

        self.start_time = Some(Instant::now());

        self.cameras
            .par_iter_mut()
            .zip(rtt_vec.par_iter())
            .for_each(|(camera, rtt)| {
                camera
                    .connection
                    .init(
                        projector_id_bit_width,
                        graycode_x_count,
                        graycode_y_count,
                        &Self::generate_timestamps(
                            projector_id_bit_width,
                            graycode_x_count,
                            graycode_y_count,
                            max_rtt + start_delay + capture_delay
                                - rtt / 2
                                - self.start_time.unwrap().elapsed().as_millis() as u32,
                            interval,
                            area_calculation_interval,
                            false,
                        ),
                    )
                    .ok();
            });

        Ok(())
    }

    /// Project an image on all projectors
    fn project_raw_all(&self, mat: &Mat) -> Result<()> {
        for id in 0..self.projector_count {
            opencv::highgui::imshow(&get_winname(id), mat)?;
        }
        wait_key(1)?;
        Ok(())
    }

    /// Show `mat` fullscreen on projector `id`'s window (no timing alignment).
    pub fn project_raw(&self, mat: &Mat, id: usize) -> Result<()> {
        opencv::highgui::imshow(&get_winname(id), mat)?;
        wait_key(1)?;
        Ok(())
    }

    /// Project the patterns used to locate each projector in the camera image
    fn process_position(&mut self) -> Result<()> {
        if let Some(start) = self.start_time {
            Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
            self.project_raw_all(&self.white_code)?;
            Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
            self.project_raw_all(&self.black_code)?;
            Ok(())
        } else {
            Err(anyhow!("Not started yet."))
        }
    }

    /// Project the projector IDs
    fn process_id(&mut self) -> Result<()> {
        if let Some(start) = self.start_time {
            for i in (0..self.projector_id_bit_width()).rev() {
                Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
                for id in 0..self.projector_count {
                    if (id >> i) & 1 == 0 {
                        imshow(&get_winname(id), &self.black_code)?;
                    } else {
                        imshow(&get_winname(id), &self.white_code)?;
                    }
                }
                Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
                for id in 0..self.projector_count {
                    if (id >> i) & 1 == 0 {
                        imshow(&get_winname(id), &self.white_code)?;
                    } else {
                        imshow(&get_winname(id), &self.black_code)?;
                    }
                }
            }
            Ok(())
        } else {
            Err(anyhow!("Not started yet."))
        }
    }

    /// Project the Gray-code patterns
    fn process_graycode(&mut self) -> Result<()> {
        if let Some(start) = self.start_time {
            let graycodes = self.graycodes.iter();
            for code in graycodes {
                Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
                self.project_raw_all(&code)?;
            }
            Self::wait_until(self.client_timestamps.pop_front().unwrap(), start)?;
            Ok(())
        } else {
            Err(anyhow!("Not started yet."))
        }
    }

    /// Projection result
    fn get_result(&mut self) -> Result<CalibrationResult> {
        if self.start_time.is_none() {
            return Err(anyhow!("Not started yet."));
        }
        if !self.client_timestamps.is_empty() {
            return Err(anyhow!("Not finished yet."));
        }

        let mut result = vec![HashMap::<CameraId, CameraResult>::new(); self.projector_count];
        for (camera_id, camera) in self.cameras.iter_mut().enumerate() {
            if let Ok(data) = camera.connection.data() {
                for (projector_id, camera_result) in data.into_iter() {
                    if projector_id >= result.len() as u32 {
                        continue;
                    }
                    result[projector_id as usize].insert(camera_id, camera_result);
                }
            }
        }

        let mut calibration_result = vec![
            CalibrationResultEntry {
                cameras: BTreeMap::<CameraId, CameraResult>::new()
            };
            self.projector_count
        ];
        for (projector_id, result) in result.iter_mut().enumerate() {
            for (camera_id_1, data_1) in result.iter() {
                calibration_result[projector_id]
                    .cameras
                    .insert(*camera_id_1, *data_1);
            }
        }

        Ok(calibration_result)
    }

    /// Calibrate (currently returns, per projector, the camera pixel positions)
    pub fn calibrate(
        &mut self,
        start_delay: u32,
        capture_delay: u32,
        interval: u32,
        area_calculation_interval: u32,
    ) -> Result<CalibrationResult> {
        for i in 0..self.projector_count {
            setup_window(i, Size::new(self.width as i32, self.height as i32))?;
        }

        self.set_timestamps(
            start_delay,
            capture_delay,
            interval,
            area_calculation_interval,
        )?;

        self.process_position()?;

        self.process_id()?;

        self.process_graycode()?;

        self.result = Some(self.get_result()?);

        Ok(self.result.as_ref().unwrap().clone())
    }

    /// Build the evenly-spaced capture schedule for the subpixel refinement
    /// pass: `additional_pattern_count` timestamps starting at `start_delay`,
    /// spaced `interval` ms apart.
    fn generate_timestamps_subpix(
        additional_pattern_count: usize,
        start_delay: u32,
        interval: u32,
    ) -> Vec<u32> {
        let mut data = vec![];
        let mut time = start_delay;

        for _ in 0..additional_pattern_count {
            data.push(time);
            time += interval;
        }

        data
    }

    /// Arm every camera for the subpixel pass: measure RTT, send each camera the
    /// scan spans (`init_subpix`) it is responsible for plus an RTT-compensated
    /// capture schedule. Returns the start instant and the host's own schedule.
    fn trigger_cameras_subpix(
        &mut self,
        spans_by_projector: &[(CoordinateSpans, CoordinateSpans)],
        start_delay: u32,
        capture_delay: u32,
        interval: u32,
    ) -> Result<(Instant, VecDeque<u32>)> {
        let rtt_vec = {
            let mut rtt_vec = vec![Option::<u32>::None; self.cameras.len()];
            self.cameras
                .par_iter_mut()
                .zip(rtt_vec.par_iter_mut())
                .for_each(|(camera, rtt)| {
                    *rtt = camera.connection.rtt(10).ok();
                });
            if rtt_vec.iter().any(|item| item.is_none()) {
                return Err(anyhow!("Failed to get all RTT"));
            }
            rtt_vec.iter().map(|item| item.unwrap()).collect::<Vec<_>>()
        };

        let max_rtt = *rtt_vec.iter().max().unwrap();

        let additional_pattern_count = spans_by_projector
            .iter()
            .map(|(x_spans, y_spans)| x_spans.max_len() + y_spans.max_len())
            .max()
            .unwrap_or_default();

        let client_timestamps = VecDeque::from_iter(Self::generate_timestamps_subpix(
            additional_pattern_count + 1,
            max_rtt + start_delay,
            interval,
        ));

        let start_time = Instant::now();

        self.cameras
            .par_iter_mut()
            .zip(rtt_vec.par_iter())
            .enumerate()
            .for_each(|(camera_id, (camera, rtt))| {
                let mut entries = vec![];
                for (x_spans, y_spans) in spans_by_projector.iter() {
                    if let Some(x_span) = x_spans.get_span_by_tag(camera_id) {
                        entries.push((
                            x_span,
                            y_spans.get_span_by_tag(camera_id).unwrap(),
                            x_spans.max_len(),
                        ));
                    } else {
                        entries.push((&(0..=0), &(0..=0), 0));
                    };
                }
                camera
                    .connection
                    .init_subpix(
                        &entries,
                        &Self::generate_timestamps_subpix(
                            additional_pattern_count,
                            max_rtt + start_delay + capture_delay
                                - rtt / 2
                                - start_time.elapsed().as_millis() as u32,
                            interval,
                        ),
                    )
                    .ok();
            });

        Ok((start_time, client_timestamps))
    }

    /// Project the per-span sweep of single scan lines (X spans first, then Y)
    /// on each projector, advancing in step with the cameras' timestamps so the
    /// refined subpixel positions can be decoded.
    fn process_subpix(
        &self,
        start: Instant,
        timestamps: &mut VecDeque<u32>,
        spans_by_projector: &[(CoordinateSpans, CoordinateSpans)],
    ) -> Result<()> {
        let max_size = spans_by_projector
            .iter()
            .map(|item| item.0.max_len() + item.1.max_len())
            .max()
            .unwrap_or_default();
        let size = Size::new(self.width as i32, self.height as i32);
        for i in 0..max_size {
            Self::wait_until(timestamps.pop_front().unwrap(), start)?;
            for (projector_id, (x_spans, y_spans)) in spans_by_projector.iter().enumerate() {
                let img;
                if i < x_spans.max_len() {
                    img = generate_axis_lines(size, &x_spans.rotate()[i], Axis::X)?;
                } else {
                    let i = i - x_spans.max_len();
                    if i < y_spans.max_len() {
                        img = generate_axis_lines(size, &y_spans.rotate()[i], Axis::Y)?;
                    } else {
                        img = generate_axis_lines(size, &[], Axis::Y)?;
                    }
                }
                self.project_raw(&img, projector_id)?;
            }
        }
        Self::wait_until(timestamps.pop_front().unwrap(), start)?;
        Ok(())
    }

    /// Refine the integer correspondences from `calibrate` to subpixel accuracy.
    ///
    /// From each projector's coarse result it derives the per-camera scan spans
    /// (grouped by `build_coordinate_spans`), sweeps single scan lines over those
    /// spans, and returns the refined projector position per camera, keyed by
    /// projector id. Must be called after `calibrate`.
    pub fn calibrate_subpix(
        &mut self,
        start_delay: u32,
        capture_delay: u32,
        interval: u32,
    ) -> Result<Vec<HashMap<usize, Point2f>>> {
        let res = self.result.as_ref().unwrap();
        let mut spans_by_projector = vec![];

        for res_by_proj in res.iter() {
            let x_pairs = res_by_proj
                .cameras
                .iter()
                .map(|(&i, item)| (item.camera_position_in_projector.x, i))
                .collect_vec();
            let y_pairs = res_by_proj
                .cameras
                .iter()
                .map(|(&i, item)| (item.camera_position_in_projector.y, i))
                .collect_vec();
            let h = 5;
            let x_spans = build_coordinate_spans(h, x_pairs);
            let y_spans = build_coordinate_spans(h, y_pairs);
            spans_by_projector.push((x_spans, y_spans));
        }

        let (start, mut ts) =
            self.trigger_cameras_subpix(&spans_by_projector, start_delay, capture_delay, interval)?;

        self.process_subpix(start, &mut ts, &spans_by_projector)?;

        let mut result = vec![HashMap::new(); self.projector_count];
        for (camera_id, camera) in self.cameras.iter_mut().enumerate() {
            let data = camera.connection.data_subpix()?;
            for (projector_id, value) in data.iter() {
                // Guard against an out-of-range id from the camera server (mirrors
                // the bounds check in `get_result`); both indexes share this length.
                if *projector_id as usize >= spans_by_projector.len() {
                    continue;
                }
                if spans_by_projector[*projector_id as usize]
                    .0
                    .get_span_by_tag(camera_id)
                    .is_none()
                {
                    continue;
                }
                result[*projector_id as usize].insert(camera_id, value.to::<f32>().unwrap());
            }
        }

        Ok(result)
    }
}

/// Groups of nearby coordinate values clustered into scan ranges, so cameras
/// that observed roughly the same projector coordinate are swept by a single
/// shared scan line during the subpixel pass.
struct CoordinateSpans {
    /// One inclusive scan range (in projector px) per cluster.
    spans: Vec<RangeInclusive<i32>>,
    /// Camera ids belonging to each span, parallel to `spans`.
    tag_groups: Vec<Vec<usize>>,
}

impl CoordinateSpans {
    /// Return, as a Vec, the value of each span at the given index
    fn get_values_at(&self, index: usize) -> Vec<Option<i32>> {
        self.spans
            .iter()
            .map(|range| {
                let value = range.start() + index as i32;
                if value <= *range.end() {
                    Some(value)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Take the i-th element of every span, in order
    /// i.e.:
    /// span_1 = [1, 2, 3]
    /// span_2 = [5, 6]
    /// res = [[1, 5], [2, 6], [3]]
    fn rotate(&self) -> Vec<Vec<i32>> {
        (0..self.max_len())
            .map(|i| self.get_values_at(i).into_iter().flatten().collect())
            .collect()
    }

    /// Return the maximum index
    fn max_len(&self) -> usize {
        self.spans
            .iter()
            .map(|range| (range.end() - range.start() + 1) as usize)
            .max()
            .unwrap_or(0)
    }

    /// Return the span for a tag
    fn get_span_by_tag(&self, tag: usize) -> Option<&RangeInclusive<i32>> {
        for (i, tags) in self.tag_groups.iter().enumerate() {
            if tags.contains(&tag) {
                return Some(&self.spans[i]);
            }
        }
        None
    }
}

/// Cluster `(value, tag)` pairs into [`CoordinateSpans`]: values within `h` of
/// each other join one cluster, and each cluster's range is padded by `h` on
/// both sides. `tag` is the camera id contributing that value.
fn build_coordinate_spans(h: i32, mut value_tag_pairs: Vec<(i32, usize)>) -> CoordinateSpans {
    if value_tag_pairs.is_empty() {
        return CoordinateSpans {
            spans: vec![],
            tag_groups: vec![],
        };
    }

    value_tag_pairs.sort_unstable_by_key(|(v, _)| *v);

    let mut spans = vec![];
    let mut tag_groups = vec![];

    let mut current_values = vec![value_tag_pairs[0].0];
    let mut current_tags = vec![value_tag_pairs[0].1];

    for i in 1..value_tag_pairs.len() {
        let (value, tag) = value_tag_pairs[i];
        if value - value_tag_pairs[i - 1].0 <= h {
            current_values.push(value);
            current_tags.push(tag);
        } else {
            let min_val = *current_values.iter().min().unwrap();
            let max_val = *current_values.iter().max().unwrap();
            spans.push((min_val - h)..=(max_val + h));
            tag_groups.push(current_tags);

            current_values = vec![value];
            current_tags = vec![tag];
        }
    }

    let min_val = *current_values.iter().min().unwrap();
    let max_val = *current_values.iter().max().unwrap();
    spans.push((min_val - h)..=(max_val + h));
    tag_groups.push(current_tags);

    CoordinateSpans { spans, tag_groups }
}
