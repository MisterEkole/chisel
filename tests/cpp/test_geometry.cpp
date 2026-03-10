// tests/cpp/test_geometry.cpp  –  Unit tests for geometry layer
#include <gtest/gtest.h>
#include "core/types.h"
#include "geometry/triangulation.h"
#include "geometry/epipolar.h"

using namespace chisel;
using namespace chisel::geometry;

TEST(Types, CameraIntrinsicsK) {
    CameraIntrinsics cam;
    cam.fx = 500; cam.fy = 500;
    cam.cx = 320; cam.cy = 240;

    Mat3 K = cam.K();
    EXPECT_DOUBLE_EQ(K(0, 0), 500);
    EXPECT_DOUBLE_EQ(K(1, 1), 500);
    EXPECT_DOUBLE_EQ(K(0, 2), 320);
    EXPECT_DOUBLE_EQ(K(1, 2), 240);
    EXPECT_DOUBLE_EQ(K(2, 2), 1);
}

TEST(Types, CameraProjectUnproject) {
    CameraIntrinsics cam;
    cam.fx = 500; cam.fy = 500;
    cam.cx = 320; cam.cy = 240;

    Vec3 p_cam(1.0, 2.0, 10.0);
    Vec2 px = cam.project(p_cam);

    EXPECT_NEAR(px.x(), 500 * 0.1 + 320, 1e-10);
    EXPECT_NEAR(px.y(), 500 * 0.2 + 240, 1e-10);

    Vec3 ray = cam.unproject(px);
    EXPECT_GT(ray.z(), 0);
}

TEST(Types, CameraPoseTransform) {
    CameraPose pose;
    pose.rotation = Quat::Identity();
    pose.translation = Vec3(1, 0, 0);

    Vec3 pw(5, 3, 2);
    Vec3 pc = pose.transform(pw);
    EXPECT_NEAR(pc.x(), 6, 1e-10);
    EXPECT_NEAR(pc.y(), 3, 1e-10);
    EXPECT_NEAR(pc.z(), 2, 1e-10);
}

TEST(Types, CameraPoseCenter) {
    CameraPose pose;
    pose.rotation = Quat::Identity();
    pose.translation = Vec3(1, 2, 3);

    Vec3 center = pose.center();
    EXPECT_NEAR(center.x(), -1, 1e-10);
    EXPECT_NEAR(center.y(), -2, 1e-10);
    EXPECT_NEAR(center.z(), -3, 1e-10);
}

TEST(Types, CameraPoseInverse) {
    CameraPose pose;
    pose.rotation = Quat(Eigen::AngleAxisd(0.1, Vec3::UnitZ()));
    pose.translation = Vec3(1, 2, 3);

    CameraPose inv = pose.inverse();
    Vec3 p(5, 3, 2);

    // Round-trip: transform → inverse transform should return original
    Vec3 p_cam = pose.transform(p);
    Vec3 p_back = inv.transform(p_cam);

    EXPECT_NEAR(p_back.x(), p.x(), 1e-10);
    EXPECT_NEAR(p_back.y(), p.y(), 1e-10);
    EXPECT_NEAR(p_back.z(), p.z(), 1e-10);
}

TEST(Triangulation, LinearTwoView) {
    CameraIntrinsics cam;
    cam.fx = 500; cam.fy = 500;
    cam.cx = 320; cam.cy = 240;

    CameraPose pose1;  // identity
    CameraPose pose2;
    pose2.translation = Vec3(1, 0, 0);  // 1m baseline

    // 3D point
    Vec3 gt(2, 1, 10);

    // Project to both cameras
    Vec3 p1 = pose1.transform(gt);
    Vec3 p2 = pose2.transform(gt);
    Vec2 px1 = cam.project(p1);
    Vec2 px2 = cam.project(p2);

    // Triangulate
    Vec3 result = triangulate_point_linear(pose1, pose2, cam, cam, px1, px2);

    EXPECT_NEAR(result.x(), gt.x(), 0.01);
    EXPECT_NEAR(result.y(), gt.y(), 0.01);
    EXPECT_NEAR(result.z(), gt.z(), 0.01);
}

TEST(Triangulation, Angle) {
    Vec3 point(0, 0, 10);
    Vec3 cam1(0, 0, 0);
    Vec3 cam2(2, 0, 0);

    double angle = triangulation_angle(point, cam1, cam2);
    EXPECT_GT(angle, 0);
    EXPECT_LT(angle, 180);
}

TEST(Epipolar, SampsonError) {
    // Identity fundamental → all points should have zero error when coplanar
    Mat3 F = Mat3::Zero();
    F(1, 2) = -1;
    F(2, 1) = 1;  // F = [0 0 0; 0 0 -1; 0 1 0] (example)

    Vec2 p1(100, 200);
    Vec2 p2(100, 200);

    double err = sampson_error(F, p1, p2);
    EXPECT_GE(err, 0);  // non-negative
}

TEST(Scene, NextPointId) {
    Scene scene;
    EXPECT_EQ(scene.next_point3d_id(), 1);

    Point3D pt;
    pt.id = 42;
    scene.points3d[42] = pt;
    EXPECT_EQ(scene.next_point3d_id(), 43);
}

TEST(Scene, NumRegistered) {
    Scene scene;
    ImageData img1, img2;
    img1.id = 1; img1.pose_valid = true;
    img2.id = 2; img2.pose_valid = false;
    scene.images[1] = img1;
    scene.images[2] = img2;
    EXPECT_EQ(scene.num_registered(), 1);
}
