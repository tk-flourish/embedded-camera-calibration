//! External camera abstraction.
//!
//! The external camera (a hand-held camera facing the calibration board) is used
//! only for the optical-center-misalignment compensation and for the conventional
//! baseline capture; the proposed method itself relies on the embedded cameras
//! (see [`crate::camera`]). To keep the core OS-independent, the backend uses
//! OpenCV's `VideoCapture`.
//!
//! The experiments were captured with a tethered Canon DSLR; that proprietary
//! backend is not included here, but the exposure settings used (ISO / aperture /
//! shutter speed — see the enums below and their call sites) are kept in the code
//! as a record of the capture conditions.

use std::sync::Mutex;

use anyhow::{bail, Result};
use opencv::{core::Mat, prelude::*, videoio};

/// Index of the default OpenCV capture device.
const DEFAULT_CAMERA_INDEX: i32 = 0;

/// DSLR capture settings used in the experiments.
///
/// These record the exposure settings of the tethered DSLR used to capture the
/// dataset. Generic backends such as [`OpenCvCamera`] ignore them (the trait
/// provides no-op defaults), since `VideoCapture` exposes no portable equivalent
/// for ISO / aperture / shutter.
#[derive(Debug, Clone, Copy)]
pub enum IsoValue {
    Auto = 0,
    Iso100 = 100,
    Iso200 = 200,
    Iso400 = 400,
    Iso800 = 800,
    Iso1600 = 1600,
    Iso3200 = 3200,
    Iso6400 = 6400,
}

#[derive(Debug, Clone, Copy)]
pub enum ApertureValue {
    F28 = 28,   // f/2.8
    F40 = 40,   // f/4.0
    F56 = 56,   // f/5.6
    F80 = 80,   // f/8.0
    F110 = 110, // f/11.0
    F160 = 160, // f/16.0
}

#[derive(Debug, Clone, Copy)]
pub enum ExposureMode {
    ProgramAE = 0,        // Program AE
    ShutterPriority = 1,  // Shutter-priority AE (Tv)
    AperturePriority = 2, // Aperture-priority AE (Av)
    Manual = 3,           // Manual exposure (M)
    Bulb = 4,             // Bulb
    FullAuto = 9,         // Full auto
}

#[derive(Debug, Clone, Copy)]
pub enum ShutterSpeed {
    S30 = 30,     // 1/30
    S60 = 60,     // 1/60
    S125 = 125,   // 1/125
    S250 = 250,   // 1/250
    S500 = 500,   // 1/500
    S1000 = 1000, // 1/1000
}

/// A camera that can capture a single still frame as an OpenCV [`Mat`].
///
/// The DSLR setting methods default to no-ops so that generic backends only need
/// to implement [`capture`](ExternalCamera::capture).
pub trait ExternalCamera {
    /// Capture a single frame.
    fn capture(&self) -> Result<Mat>;

    /// Set the exposure mode (manual/auto). No-op unless overridden.
    fn set_exposure_mode(&self, _mode: ExposureMode) -> Result<()> {
        Ok(())
    }
    /// Set the ISO sensitivity. No-op unless overridden.
    fn set_iso(&self, _iso: IsoValue) -> Result<()> {
        Ok(())
    }
    /// Set the aperture (f-number). No-op unless overridden.
    fn set_aperture(&self, _aperture: ApertureValue) -> Result<()> {
        Ok(())
    }
    /// Set the shutter speed. No-op unless overridden.
    fn set_shutter_speed(&self, _speed: ShutterSpeed) -> Result<()> {
        Ok(())
    }
    /// Run autofocus once and lock it. No-op unless overridden.
    fn auto_focus_and_lock(&self) -> Result<()> {
        Ok(())
    }
}

/// OS-independent backend backed by OpenCV `VideoCapture` (the default).
///
/// The capture device is wrapped in a [`Mutex`] because `VideoCapture::read`
/// needs `&mut self`, while [`ExternalCamera::capture`] takes `&self`.
pub struct OpenCvCamera {
    capture: Mutex<videoio::VideoCapture>,
}

impl OpenCvCamera {
    /// Open the default capture device.
    pub fn new() -> Result<Self> {
        Self::with_index(DEFAULT_CAMERA_INDEX)
    }

    /// Open the capture device at `index`.
    pub(crate) fn with_index(index: i32) -> Result<Self> {
        let capture = videoio::VideoCapture::new(index, videoio::CAP_ANY)?;
        if !capture.is_opened()? {
            bail!("failed to open camera device {index}");
        }
        Ok(Self {
            capture: Mutex::new(capture),
        })
    }
}

impl ExternalCamera for OpenCvCamera {
    fn capture(&self) -> Result<Mat> {
        let mut capture = self.capture.lock().expect("external camera mutex poisoned");
        let mut frame = Mat::default();
        capture.read(&mut frame)?;
        if frame.empty() {
            bail!("captured an empty frame");
        }
        Ok(frame)
    }
}

/// Construct the default external-camera backend ([`OpenCvCamera`]).
pub fn default_external_camera() -> Result<Box<dyn ExternalCamera>> {
    Ok(Box::new(OpenCvCamera::new()?))
}
