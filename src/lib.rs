#[cfg(test)]
mod generators;
#[cfg(test)]
mod viewer;

use eqsolver::multivariable::GaussNewtonFD;
use nalgebra::{DVector, Point, Point3, RealField, SVector, SimdValue, UnitVector3, Vector3};
use std::fmt::Debug;
pub trait Scalar: Clone + PartialEq + Debug + SimdValue + RealField + Copy + 'static {}
impl<T: Clone + PartialEq + Debug + SimdValue + RealField + Copy + 'static> Scalar for T {}

#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct Plane<T: Scalar> {
    position: Point3<T>,
    normal: UnitVector3<T>,
}

impl<T: Scalar> Fittable<T, 3, 3> for Plane<T> {
    fn from_raw(variables: SVector<T, 3>) -> Self {
        let theta = variables.x;
        let phi = variables.y;
        let d = variables.z;

        let normal = UnitVector3::new_normalize(Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        ));

        Plane {
            position: (normal.into_inner() * d).into(),
            normal,
        }
    }

    fn into_raw(&self) -> SVector<T, 3> {
        let x = self.normal.x;
        let y = self.normal.y;
        let z = self.normal.z;
        let theta = z.acos();
        let phi = y.signum() * (x / (x * x + y * y).sqrt()).acos();
        let d = self.position.coords.dot(&self.normal);

        Vector3::new(theta, phi, d)
    }

    fn distance_to(&self, point: Point<T, 3>) -> T {
        (self.position - point).dot(&self.normal)
    }
}

pub trait Fittable<T: Scalar, const R: usize, const C: usize> {
    fn from_raw(variables: SVector<T, R>) -> Self;
    fn into_raw(&self) -> SVector<T, R>;
    fn distance_to(&self, point: Point<T, C>) -> T;
}

pub fn fit_least_squares<const R: usize, const C: usize, G: Fittable<f32, R, C>>(
    points: &[Point<f32, C>],
    guess: &G,
) -> G {
    G::from_raw(
        GaussNewtonFD::new(|inputs: SVector<f32, R>| {
            let geometry = G::from_raw(inputs);
            DVector::from_iterator(
                points.len(),
                points
                    .iter()
                    .copied()
                    .map(|point| geometry.distance_to(point)),
            )
        })
        .with_tol(points.len() as f32 * 10.0_f32)
        .solve(guess.into_raw())
        .unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use macroquad::prelude::*;
    use nalgebra::Vector3;

    use crate::{
        generators::{NoiseAlongZ, OffsetInWorldSpace, ShapeFunction3D, sample_random},
        viewer::Viewer,
    };

    use super::*;

    #[test]
    fn test_plane_sample_random() {
        let plane = Plane {
            position: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::x_axis(),
        };

        let measurements = sample_random(
            &plane
                .clone()
                .modify(OffsetInWorldSpace(100.0 * *plane.normal))
                .modify(NoiseAlongZ { amplitude: 5.0 }),
            -25.0..25.0,
            -25.0..25.0,
            1000,
        );

        let fitted = fit_least_squares(
            &measurements.iter().map(|x| x.position).collect::<Vec<_>>(),
            &plane,
        );

        println!("{:?}", fitted.position - plane.position);

        Viewer::new()
            .add(plane.with_color(color_u8!(255, 255, 255, 150)))
            .add(fitted.with_color(color_u8!(255, 0, 255, 150)))
            .add(measurements)
            .show();
    }
}
