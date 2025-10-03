use macroquad::prelude::*;
use nalgebra::{Point3, UnitQuaternion, UnitVector3, Vector3};

use crate::{Plane, generators::OrientedPoint};

#[derive(Default)]
pub struct Viewer(Vec<Box<dyn Drawable>>);

impl Viewer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, viewable: impl Drawable + 'static) -> Self {
        self.0.push(Box::new(viewable));
        self
    }

    pub fn show(self) {
        macroquad::Window::new("Measurement Viewer", async move {
            let mut camera = Camera3D {
                position: vec3(-100., 0., 0.),
                up: vec3(0., 1., 0.),
                target: vec3(0., 0., 0.),
                projection: Projection::Perspective,
                ..Default::default()
            };

            loop {
                clear_background(BLACK);

                let delta = mouse_delta_position();
                let mut eye_direction = (camera.target - camera.position).normalize();
                let camera_sideways = camera.up.cross(eye_direction);

                if is_mouse_button_down(MouseButton::Left) {
                    let mut rotation = Quat::IDENTITY;
                    rotation *= Quat::from_axis_angle(camera.up, -delta.x);
                    rotation *= Quat::from_axis_angle(camera_sideways, delta.y);
                    camera.up = rotation * camera.up;
                    eye_direction = rotation * eye_direction;
                } else if is_mouse_button_down(MouseButton::Right) {
                    camera.position += camera_sideways * -delta.x * 100.0;
                    camera.position += camera.up * -delta.y * 100.0;
                }

                const CAMERA_SPEED: f32 = 1.0;
                if is_key_down(KeyCode::W) {
                    camera.position += eye_direction * CAMERA_SPEED;
                }
                if is_key_down(KeyCode::S) {
                    camera.position -= eye_direction * CAMERA_SPEED;
                }
                if is_key_down(KeyCode::A) {
                    camera.position += camera_sideways * CAMERA_SPEED;
                }
                if is_key_down(KeyCode::D) {
                    camera.position -= camera_sideways * CAMERA_SPEED;
                }
                if is_key_down(KeyCode::Q) {
                    camera.position += camera.up * CAMERA_SPEED;
                }
                if is_key_down(KeyCode::E) {
                    camera.position -= camera.up * CAMERA_SPEED;
                }

                camera.target = camera.position + eye_direction;

                set_camera(&camera);

                self.0.iter().for_each(|drawable| drawable.draw());

                next_frame().await;
            }
        });
    }
}

pub(crate) fn vec3_from_point3(point: Point3<f32>) -> Vec3 {
    Vec3::new(point.x, point.y, point.z)
}

pub(crate) fn vec3_from_vector3(vector: UnitVector3<f32>) -> Vec3 {
    Vec3::new(vector.x, vector.y, vector.z)
}

pub trait Drawable {
    fn draw(&self);
}

impl Drawable for Vec<OrientedPoint<f32>> {
    fn draw(&self) {
        self.iter().for_each(|point| {
            draw_cube(
                vec3_from_point3(point.position),
                Vec3::new(0.25, 0.25, 0.25),
                None,
                WHITE,
            );
        });
    }
}

pub fn draw_quad(vertices: [Vertex; 4], texture: Option<Texture2D>) {
    draw_mesh(&Mesh {
        vertices: vertices.to_vec(),
        indices: [0, 1, 2, 0, 2, 3].to_vec(),
        texture,
    });
}

impl Drawable for Plane<f32> {
    fn draw(&self) {
        let rotation =
            UnitQuaternion::rotation_between_axis(&Vector3::z_axis(), &self.normal).unwrap();

        let extent = vec2(25.0, 25.0);
        let primary = extent.x * vec3_from_vector3(rotation * Vector3::x_axis());
        let secondary = extent.y * vec3_from_vector3(rotation * Vector3::y_axis());
        let position = vec3_from_point3(self.position);
        let color = color_u8!(255, 255, 255, 150);

        draw_quad(
            [
                Vertex::new2(position - primary - secondary, vec2(0., 0.), color),
                Vertex::new2(position - primary + secondary, vec2(0., 1.), color),
                Vertex::new2(position + primary + secondary, vec2(1., 1.), color),
                Vertex::new2(position + primary - secondary, vec2(1., 0.), color),
            ],
            None
        );
    }
}
