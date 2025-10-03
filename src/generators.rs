use std::{marker::PhantomData, ops::Add};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rand::{
    Rng,
    distr::uniform::{SampleRange, SampleUniform},
};

use crate::{Plane, Scalar};

#[derive(Clone, Debug, PartialEq)]
pub struct OrientedPoint<T: Scalar> {
    pub position: Point3<T>,
    pub rotation: UnitQuaternion<T>,
}

impl<T: Scalar> Add<Vector3<T>> for OrientedPoint<T> {
    type Output = OrientedPoint<T>;

    fn add(self, rhs: Vector3<T>) -> Self::Output {
        OrientedPoint {
            position: self.position + self.rotation.clone() * rhs,
            rotation: self.rotation,
        }
    }
}

pub trait ShapeFunction3D<T: Scalar> {
    fn sample(&self, u: T, v: T) -> OrientedPoint<T>;

    fn modify(self, modifier: impl ShapeModifier3D<T>) -> impl ShapeFunction3D<T>
    where
        Self: Sized,
    {
        ModifiedShapeFunction3D {
            inner: self,
            modifier,
            phantom: PhantomData,
        }
    }
}

pub struct ModifiedShapeFunction3D<T: Scalar, F: ShapeFunction3D<T>, M: ShapeModifier3D<T>> {
    inner: F,
    modifier: M,
    phantom: PhantomData<T>,
}

impl<T: Scalar, F: ShapeFunction3D<T>, M: ShapeModifier3D<T>> ShapeFunction3D<T>
    for ModifiedShapeFunction3D<T, F, M>
{
    fn sample(&self, u: T, v: T) -> OrientedPoint<T> {
        self.modifier
            .modify(u.clone(), v.clone(), self.inner.sample(u, v))
    }
}

pub trait ShapeModifier3D<T: Scalar> {
    fn modify(&self, u: T, v: T, point: OrientedPoint<T>) -> OrientedPoint<T>;
}

pub fn sample_random<T: Scalar + SampleUniform>(
    shape_function: &impl ShapeFunction3D<T>,
    u: impl SampleRange<T> + Clone,
    v: impl SampleRange<T> + Clone,
    count: u32,
) -> Vec<OrientedPoint<T>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            let u = rng.random_range(u.clone());
            let v = rng.random_range(v.clone());
            shape_function.sample(u, v)
        })
        .collect()
}

pub struct NoiseAlongZ<T: Scalar> {
    pub amplitude: T,
}

impl<T: Scalar + SampleUniform + Copy> ShapeModifier3D<T> for NoiseAlongZ<T> {
    fn modify(&self, _: T, _: T, point: OrientedPoint<T>) -> OrientedPoint<T> {
        point + (Vector3::z() * rand::rng().random_range(-self.amplitude..self.amplitude))
    }
}

impl<T: Scalar> ShapeFunction3D<T> for Plane<T> {
    fn sample(&self, u: T, v: T) -> OrientedPoint<T> {
        OrientedPoint {
            position: self.position.clone(),
            rotation: UnitQuaternion::rotation_between_axis(&Vector3::z_axis(), &self.normal)
                .unwrap(),
        } + Vector3::new(u, v, T::zero())
    }
}
