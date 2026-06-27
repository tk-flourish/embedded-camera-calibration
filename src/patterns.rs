//! Generation of projected pattern images (checkerboards, axis lines).

use anyhow::Result;
use opencv::{core::*, imgproc::*};

/// Generate a checkerboard image with `stride`-pixel squares on a black background.
/// Filled squares use `color`; the alternating squares are left black.
pub fn generate_checkered(height: i32, width: i32, color: Scalar, stride: i32) -> Result<Mat> {
    let mut target =
        Mat::new_size_with_default(Size::new(width, height), CV_8UC3, Scalar::default())?;
    for x in 0..(width / stride) {
        for y in 0..(height / stride) {
            if (x + y) % 2 == 0 {
                rectangle(
                    &mut target,
                    Rect::new(x * stride, y * stride, stride, stride),
                    color,
                    FILLED,
                    LINE_8,
                    0,
                )?;
            }
        }
    }
    Ok(target)
}

/// Which axis a set of line positions is specified along.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum Axis {
    /// Positions are x-coordinates; lines are vertical.
    X,
    /// Positions are y-coordinates; lines are horizontal.
    Y,
}

/// Draw white full-span lines on black at each position in `values`.
/// `specified_by` selects whether the values are x (vertical lines) or y (horizontal lines).
pub fn generate_axis_lines(size: Size, values: &[i32], specified_by: Axis) -> Result<Mat> {
    let white = Scalar::new(255., 255., 255., 0.);
    let black = Scalar::default();
    let mut target = Mat::new_size_with_default(size, CV_8UC3, black)?;

    match specified_by {
        Axis::X => {
            for &x in values {
                line(
                    &mut target,
                    Point::new(x, 0),
                    Point::new(x, size.height - 1),
                    white,
                    1,
                    LINE_8,
                    0,
                )?;
            }
        }
        Axis::Y => {
            for &y in values {
                line(
                    &mut target,
                    Point::new(0, y),
                    Point::new(size.width - 1, y),
                    white,
                    1,
                    LINE_8,
                    0,
                )?;
            }
        }
    }

    Ok(target)
}
