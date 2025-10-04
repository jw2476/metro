#[cfg(test)]
mod generators;
#[cfg(test)]
mod viewer;

use nalgebra::{
    Const, DMatrix, DVector, Dyn, MatrixXx3, OMatrix, Point, Point3, RealField, SMatrix, SVector,
    SimdValue, UnitVector3, Vector3,
};
use std::{fmt::Debug, thread::current};
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

pub struct FitResult<T: Scalar, G> {
    pub residuals: DVector<T>,
    pub geometry: G,
}

fn gauss_newton<const R: usize, const C: usize, G: Fittable<f32, R, C>>(
    points: &[Point<f32, C>],
    mut guess: G,
) -> FitResult<f32, G> {
    let step = f32::EPSILON.sqrt();
    let tolerance = 1.0e-3;

    let get_residuals_from = |guess: &G| -> DVector<f32> {
        DVector::from_iterator(
            points.len(),
            points.iter().map(|point| guess.distance_to(*point)),
        )
    };

    let mut current_residuals = DVector::repeat(points.len(), 0.0);
    let mut guess_delta = SVector::<f32, R>::repeat(f32::MAX);
    let mut jacobian = OMatrix::<f32, Dyn, Const<R>>::repeat(points.len(), 0.0);

    while guess_delta.magnitude() > tolerance {
        current_residuals = get_residuals_from(&guess);

        let jcols = jacobian.ncols();
        for i in 0..jcols {
            let guess_with_stepped_row = {
                let mut raw_guess = guess.into_raw();
                raw_guess[i] += step;
                G::from_raw(raw_guess)
            };
            let residuals_with_stepped_row = get_residuals_from(&guess_with_stepped_row);
            let residuals_derivative_for_row =
                (residuals_with_stepped_row - &current_residuals) / step;
            jacobian.set_column(i, &residuals_derivative_for_row);
        }

        let transposed_jacobian = jacobian.transpose();
        if let Some(x) = (&transposed_jacobian * &jacobian).try_inverse() {
            let jacobian_pseudoinverse = x * transposed_jacobian;
            guess_delta = jacobian_pseudoinverse * &current_residuals;
            guess = G::from_raw(guess.into_raw() - guess_delta);
        } else {
            todo!()
        }
    }

    FitResult {
        residuals: current_residuals,
        geometry: guess,
    }
}

pub fn fit_least_squares<const R: usize, const C: usize, G: Fittable<f32, R, C>>(
    points: &[Point<f32, C>],
    guess: G,
) -> FitResult<f32, G> {
    gauss_newton(points, guess)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

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
                .modify(NoiseAlongZ { amplitude: 0.1 }),
            -25.0..25.0,
            -25.0..25.0,
            1_000_000,
        );

        let points = measurements.iter().map(|x| x.position).collect::<Vec<_>>();

        let start = Instant::now();
        let fitted = fit_least_squares(&points, plane.clone());
        for _ in 0..99 {
            let fitted = fit_least_squares(&points, plane.clone());
        }
        println!("{}", (Instant::now() - start).as_secs_f32() / 100.0);

        println!(
            "Fit displacement: {:?}",
            fitted.geometry.position - plane.position
        );

        /*
        Viewer::new()
            .add(plane.with_color(color_u8!(255, 255, 255, 150)))
            .add(fitted.geometry.with_color(color_u8!(255, 0, 255, 150)))
            .show();*/
    }
}
