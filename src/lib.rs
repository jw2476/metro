#[cfg(test)]
mod generators;
#[cfg(test)]
mod viewer;

use nalgebra::{Point3, RealField, SimdValue, UnitVector3};
use std::fmt::Debug;
pub trait Scalar: Clone + PartialEq + Debug + SimdValue + RealField + 'static {}
impl<T: Clone + PartialEq + Debug + SimdValue + RealField + 'static> Scalar for T {}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct Plane<T: Scalar> {
    position: Point3<T>,
    normal: UnitVector3<T>,
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{
        generators::{NoiseAlongZ, ShapeFunction3D, sample_random},
        viewer::Viewer,
    };

    use super::*;

    #[test]
    fn test_plane_sample_random() {
        let plane = Plane {
            position: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::x_axis(),
        };

        Viewer::new()
            .add(sample_random(
                &plane.clone().modify(NoiseAlongZ { amplitude: 5.0 }),
                -25.0..25.0,
                -25.0..25.0,
                1000,
            ))
            .add(plane)
            .show();
    }
}
