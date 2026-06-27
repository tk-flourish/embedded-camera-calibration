//! TCP client for the embedded-camera ("picamera") servers.
//!
//! Each camera runs a Python server that decodes Gray-code patterns and reports
//! the resulting projector/camera correspondences. Communication uses a simple
//! length-prefixed binary protocol: every frame on the wire is a big-endian
//! `u32` byte length followed by that many payload bytes. A request payload is a
//! length-prefixed command string followed by the command arguments; a response
//! payload is a length-prefixed status string ("OK" on success) followed by the
//! returned data. All integers and floats on the wire are big-endian.

use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpStream, ToSocketAddrs},
    ops::RangeInclusive,
    time::Instant,
};

use anyhow::{anyhow, bail, Result};
use opencv::{
    core::{Mat, Point2d, Point2i},
    imgcodecs,
};

use crate::types::*;

/// A decoded server response: the status string plus the raw payload bytes.
#[derive(Debug)]
struct Response {
    /// Status string sent by the server; "OK" indicates success.
    status: String,
    /// Command-specific payload following the status field (may be empty).
    data: Vec<u8>,
}

impl Response {
    /// Succeed only when the server reported the "OK" status.
    fn validate_ok(&self) -> Result<()> {
        if self.status == "OK" {
            Ok(())
        } else {
            Err(anyhow!(format!("Status is not OK, but {}.", self.status)))
        }
    }
}

/// An open TCP connection to a single embedded-camera server.
pub struct CameraConnection {
    stream: TcpStream,
}

impl CameraConnection {
    /// Open a connection to the camera server at `address`.
    pub fn new<A>(address: A) -> io::Result<Self>
    where
        A: ToSocketAddrs,
    {
        let stream = TcpStream::connect(address)?;
        Ok(CameraConnection { stream })
    }

    /// Send one length-prefixed frame: a big-endian `u32` length then `data`.
    fn send_raw(&mut self, data: &[u8]) -> Result<()> {
        let mut buffer = Vec::<u8>::new();
        buffer.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buffer.extend_from_slice(data);
        self.stream.write_all(&buffer)?;
        Ok(())
    }

    /// Send a command request: a length-prefixed command string then `data`.
    /// The whole request is wrapped in one outer length-prefixed frame.
    pub(crate) fn send(&mut self, command: &str, data: &[u8]) -> Result<()> {
        let mut send_data: Vec<u8> = vec![];
        let command_bytes = command.as_bytes();
        send_data.extend((command_bytes.len() as u32).to_be_bytes());
        send_data.extend(command_bytes);
        send_data.extend(data);
        self.send_raw(&send_data)
    }

    /// Read one response frame and split it into status string and payload.
    /// Layout: big-endian `u32` status length, that many UTF-8 status bytes,
    /// then the remaining bytes as the payload.
    fn read(&mut self) -> Result<Response> {
        let raw_data = self.read_raw()?;
        if raw_data.len() < 4 {
            bail!("response shorter than the 4-byte status length header");
        }
        let status_len = u32::from_be_bytes(raw_data[0..4].try_into().unwrap()) as usize;
        if raw_data.len() < status_len + 4 {
            bail!(
                "response truncated: status length {status_len} exceeds the {} payload bytes",
                raw_data.len() - 4
            );
        }
        let status = String::from_utf8(raw_data[4..(status_len + 4)].to_vec())?;
        let data = raw_data[(status_len + 4)..].to_vec();
        Ok(Response { status, data })
    }

    /// Read one length-prefixed frame: a big-endian `u32` length then that many
    /// bytes. Blocks until the full frame has been received.
    fn read_raw(&mut self) -> Result<Vec<u8>> {
        let mut length = [0u8; 4];
        self.stream.read_exact(&mut length)?;
        let length = u32::from_be_bytes(length);
        let mut data = vec![0; length as usize];
        self.stream.read_exact(&mut data)?;
        Ok(data)
    }

    /// Measure the average round-trip time over `trial` "RTT" pings, in ms.
    pub fn rtt(&mut self, trial: u32) -> Result<u32> {
        if trial == 0 {
            bail!("rtt trial count must be greater than zero");
        }
        let mut sum = 0;
        for _ in 0..trial {
            let start = Instant::now();
            self.send("RTT", &[])?;
            let response = self.read()?;
            let rtt = start.elapsed().as_millis();
            response.validate_ok()?;
            sum += rtt as u32;
        }
        Ok(sum / trial)
    }

    /// Configure the camera for Gray-code decoding via the "INIT" command.
    /// Payload: `id_bits_count`, `graycode_x_count`, `graycode_y_count` each as a
    /// big-endian `u32`, followed by one `u32` capture timestamp per pattern.
    pub fn init(
        &mut self,
        id_bits_count: u32,
        graycode_x_count: u32,
        graycode_y_count: u32,
        timestamps: &[u32],
    ) -> Result<()> {
        let mut data = vec![];
        data.extend(id_bits_count.to_be_bytes());
        data.extend(graycode_x_count.to_be_bytes());
        data.extend(graycode_y_count.to_be_bytes());
        for time in timestamps {
            data.extend(time.to_be_bytes());
        }
        self.send("INIT", &data)?;
        let response = self.read()?;
        response.validate_ok()
    }

    /// Fetch the integer (pixel-resolution) decoded correspondences via "DATA".
    /// Returns a map from projector ID to the camera/projector pixel positions.
    pub fn data(&mut self) -> Result<HashMap<ProjectorId, CameraResult>> {
        self.send("DATA", &[])?;
        let response = self.read()?;
        response.validate_ok()?;

        // Each record is 5 big-endian u32 fields: id (projector m), x, y
        // (projector pixel p_m(n)), center_x, center_y (centroid camera pixel
        // c_n(m) that received projector m's light).
        let record_size = 4 * 5;
        if response.data.len() % record_size != 0 {
            bail!(
                "DATA payload of {} bytes is not a multiple of the {record_size}-byte record",
                response.data.len()
            );
        }
        let record_count = response.data.len() / record_size;

        // Read the `field_index`-th u32 (each 4 bytes) of record `record_index`.
        let decode_u32 = |record_index: usize, field_index: usize| {
            let offset = record_index * record_size + field_index * 4;
            u32::from_be_bytes(response.data[offset..(offset + 4)].try_into().unwrap())
        };

        let mut result = HashMap::<ProjectorId, CameraResult>::new();
        for i in 0..record_count {
            let id = decode_u32(i, 0);
            let x = decode_u32(i, 1) as i32;
            let y = decode_u32(i, 2) as i32;
            let center_x = decode_u32(i, 3) as i32;
            let center_y = decode_u32(i, 4) as i32;
            result.insert(
                id,
                CameraResult {
                    camera_position_in_projector: Point2i::new(x, y),
                    projector_position_in_camera: Point2i::new(center_x, center_y),
                },
            );
        }
        Ok(result)
    }

    /// Capture a still image via "CAPTURE" and decode it to a color `Mat`.
    /// The payload is an encoded image (e.g. JPEG/PNG) decoded with `imdecode`.
    pub fn capture(&mut self) -> Result<Mat> {
        self.send("CAPTURE", &[])?;
        let response = self.read()?;
        response.validate_ok()?;
        let image =
            imgcodecs::imdecode(&Mat::from_slice(&response.data)?, imgcodecs::IMREAD_COLOR)?;
        Ok(image)
    }

    /// Configure subpixel refinement via "INIT_SUBPIX".
    /// Payload: span count (`u32`); per span the inclusive x range start/end and
    /// y range start/end (four `i32`s) plus `y_starts_on` as a `u32`; then the
    /// timestamp count (`u32`) followed by one `u32` per capture timestamp.
    pub fn init_subpix(
        &mut self,
        spans: &[(&RangeInclusive<i32>, &RangeInclusive<i32>, usize)],
        timestamps: &[u32],
    ) -> Result<()> {
        let mut data = vec![];
        data.extend((spans.len() as u32).to_be_bytes());
        for &(x_span, y_span, y_starts_on) in spans.iter() {
            data.extend(x_span.start().to_be_bytes());
            data.extend(x_span.end().to_be_bytes());
            data.extend(y_span.start().to_be_bytes());
            data.extend(y_span.end().to_be_bytes());
            data.extend((y_starts_on as u32).to_be_bytes());
        }
        data.extend((timestamps.len() as u32).to_be_bytes());
        for time in timestamps {
            data.extend(time.to_be_bytes());
        }
        self.send("INIT_SUBPIX", &data)?;
        let response = self.read()?;
        response.validate_ok()
    }

    /// Fetch the subpixel-refined projector positions via "DATA_SUBPIX".
    /// Returns a map from projector ID to subpixel position; records whose
    /// coordinates are NaN (refinement failed) are skipped.
    pub fn data_subpix(&mut self) -> Result<HashMap<ProjectorId, Point2d>> {
        self.send("DATA_SUBPIX", &[])?;
        let response = self.read()?;
        response.validate_ok()?;

        // Each record is a u32 id (4 bytes) then two f64 coords (8 bytes each).
        let record_size = 4 + 8 * 2;
        if response.data.len() % record_size != 0 {
            bail!(
                "DATA_SUBPIX payload of {} bytes is not a multiple of the {record_size}-byte record",
                response.data.len()
            );
        }
        let record_count = response.data.len() / record_size;

        // Read a u32 at `field_offset` bytes into record `record_index`.
        let decode_u32 = |record_index: usize, field_offset: usize| {
            let offset = record_index * record_size + field_offset;
            u32::from_be_bytes(response.data[offset..(offset + 4)].try_into().unwrap())
        };

        // Read an f64 at `field_offset` bytes into record `record_index`.
        let decode_f64 = |record_index: usize, field_offset: usize| {
            let offset = record_index * record_size + field_offset;
            f64::from_be_bytes(response.data[offset..(offset + 8)].try_into().unwrap())
        };

        let mut result = HashMap::<ProjectorId, Point2d>::new();
        for i in 0..record_count {
            let id = decode_u32(i, 0);
            let x = decode_f64(i, 4);
            let y = decode_f64(i, 12);
            if !x.is_nan() && !y.is_nan() {
                result.insert(id, Point2d::new(x, y));
            }
        }
        Ok(result)
    }
}
