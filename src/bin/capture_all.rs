//! Capture one frame from every embedded camera and save them as PNGs,
//! one file per camera (`0.png` .. `3.png`).
//!
//! By default frames are written to a fresh timestamped directory under
//! `.res/capture_all/`; pass `--out <DIR>` to write to a specific directory instead.

use std::{fs, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use embedded_camera_calibration::{camera::CameraConnection, debug_viz::timestamped_output_dir};
use opencv::{core::Vector, imgcodecs::imwrite};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[derive(Parser)]
#[command(about = "Capture one frame from every embedded camera")]
struct Args {
    /// Output directory for the captured frames. Defaults to a fresh
    /// timestamped directory under `.res/capture_all/`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Embedded-camera server addresses (host:port), one per camera in camera-id
    /// order. Defaults to the prototype rig.
    #[arg(long, num_args = 4, default_values = [
        "192.168.0.101:58919",
        "192.168.0.102:58919",
        "192.168.0.103:58919",
        "192.168.0.104:58919",
    ])]
    cameras: Vec<String>,
}

/// Entry point: capture one frame from each embedded camera into the output directory.
fn main() -> Result<()> {
    // ---- parse CLI args ----
    let args = Args::parse();

    // ---- prepare output directory ----
    // Default to a fresh timestamped directory under .res/capture_all/.
    let path = args
        .out
        .unwrap_or_else(|| timestamped_output_dir(".res/capture_all"));

    // Create the directory recursively
    fs::create_dir_all(&path)?;

    // ---- capture & save from every camera (in parallel) ----
    args.cameras
        .par_iter()
        .enumerate()
        .try_for_each(|(i, addr)| -> Result<()> {
            let img = CameraConnection::new(addr)?.capture()?;
            imwrite(
                path.join(format!("{i}.png")).to_str().unwrap(),
                &img,
                &Vector::default(),
            )?;
            Ok(())
        })?;

    Ok(())
}
