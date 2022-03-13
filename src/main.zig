const std = @import("std");
pub const log_level: std.log.Level = .info;
const sqrt = std.math.sqrt;

const Vec3f = @Vector(3, f32);
const WorkQueue = std.atomic.Queue(RenderArgs);

var rand: std.rand.DefaultPrng = undefined;

fn randf32(min: f32, max: f32) f32 {
    return min + rand.random().float(f32) * (max - min);
}

const Ray = struct {
    origin: Vec3f,
    direction: Vec3f,

    fn new(origin: Vec3f, direction: Vec3f) Ray {
        return .{ .origin = origin, .direction = direction };
    }

    fn at(self: Ray, t: f32) Vec3f {
        return self.origin + self.direction * @splat(3, t);
    }
};

const HitRecord = struct {
    point: Vec3f,
    normal: Vec3f,
    t: f32,
    front_face: bool,
    material: Material,
};

const Material = union(enum) {
    lambertian: struct { albedo: Vec3f },
    metal: struct {
        albedo: Vec3f,
        fuzz: f32,
    },
    dielectric: struct { eta: f32 },

    /// Scatters an incoming ray based on the material.
    /// Returns null if the scatterd ray is absorbed instead.
    fn scatter(self: Material, ray: Ray, rec: HitRecord, attenuation: *Vec3f) ?Ray {
        switch (self) {
            .lambertian => |mat| {
                var scatter_direction = rec.normal + vec3fNormalize(vec3fRandUnitSphere());
                if (vec3fNearZero(scatter_direction)) // degenerate scatter direction
                    scatter_direction = rec.normal;
                const scattered = Ray.new(rec.point, scatter_direction);
                attenuation.* = mat.albedo;
                return scattered;
            },
            .metal => |mat| {
                const reflected = vec3fReflect(vec3fNormalize(ray.direction), rec.normal);
                const scattered = Ray.new(rec.point, reflected + vec3fSplat(mat.fuzz) * vec3fRandUnitSphere());
                attenuation.* = mat.albedo;
                return if (vec3fDot(scattered.direction, rec.normal) > 0) scattered else null;
            },
            .dielectric => |mat| {
                attenuation.* = vec3fSplat(1);
                // 1.0     eta
                // air <-> material
                const eta_ratio = if (rec.front_face) (1.0 / mat.eta) else mat.eta;

                const unit_direction = vec3fNormalize(ray.direction);
                const cos_theta = std.math.min(-vec3fDot(unit_direction, rec.normal), 1.0);
                const sin_theta = sqrt(1.0 - cos_theta * cos_theta);

                // Schlick approximation for reflectance
                const r0 = (1 - eta_ratio) / (1 + eta_ratio);
                const r1 = r0 * r0;
                const reflectance = r1 + (1 - r1) * std.math.pow(f32, (1 - cos_theta), 5);

                // Total internal reflection
                const scattered = if (eta_ratio * sin_theta > 1.0 or reflectance > rand.random().float(f32))
                    vec3fReflect(unit_direction, rec.normal)
                else
                    vec3fRefract(unit_direction, rec.normal, eta_ratio);
                return Ray.new(rec.point, scattered);
            },
        }
    }

    fn lambertian(albedo: Vec3f) Material {
        return .{ .lambertian = .{ .albedo = albedo } };
    }

    fn metal(albedo: Vec3f, fuzz: f32) Material {
        return .{ .metal = .{ .albedo = albedo, .fuzz = fuzz } };
    }

    fn dielectric(eta: f32) Material {
        return .{ .dielectric = .{ .eta = eta } };
    }
};

const Sphere = struct {
    center: Vec3f,
    radius: f32,
    material: Material,

    fn hit(self: Sphere, ray: Ray, bound: struct { t_min: f32 = std.math.f32_min, t_max: f32 = std.math.f32_max }) ?HitRecord {
        const oc = ray.origin - self.center;
        const a = vec3fDot(ray.direction, ray.direction);
        const half_b = vec3fDot(oc, ray.direction);
        const c = vec3fDot(oc, oc) - self.radius * self.radius;

        // Quadratic equation!
        const discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return null;
        const sqrtd = sqrt(discriminant);

        // Test both roots
        var root = (-half_b - sqrtd) / a;
        if (root < bound.t_min or bound.t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < bound.t_min or bound.t_max < root) {
                return null;
            }
        }

        const p = ray.at(root);
        const outward_normal = (p - self.center) / vec3fSplat(self.radius);
        const front_face = vec3fDot(ray.direction, outward_normal) < 0;
        return HitRecord{
            .point = p,
            .normal = if (front_face) outward_normal else -outward_normal,
            .t = root,
            .front_face = front_face,
            .material = self.material,
        };
    }
};

const World = struct {
    allocator: std.mem.Allocator,
    spheres: std.ArrayList(Sphere),

    fn init(allocator: std.mem.Allocator) !World {
        return World{
            .allocator = allocator,
            .spheres = std.ArrayList(Sphere).init(allocator),
        };
    }

    fn deinit(self: *World) void {
        self.spheres.deinit();
    }

    fn add(self: *World, sphere: Sphere) !void {
        try self.spheres.append(sphere);
    }

    fn hit(self: World, ray: Ray, bound: struct { t_min: f32 = std.math.f32_min, t_max: f32 = std.math.f32_max }) ?HitRecord {
        var final_rec: ?HitRecord = null;
        var closest = bound.t_max;
        for (self.spheres.items) |sphere| {
            if (sphere.hit(ray, .{ .t_min = bound.t_min, .t_max = closest })) |rec| {
                final_rec = rec;
                closest = rec.t;
            }
        }
        return final_rec;
    }
};

const Camera = struct {
    origin: Vec3f,
    horizontal: Vec3f,
    vertical: Vec3f,
    lower_left_corner: Vec3f,
    lens_radius: f32,
    w: Vec3f,
    u: Vec3f,
    v: Vec3f,

    fn init(
        origin: Vec3f, // location of camera
        look_pos: Vec3f, // position which the camera points at
        up: Vec3f, // the "up" direction of the camera
        vfov: f32, // vertical field of view in degrees
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) Camera {
        const vfov_rad = vfov * std.math.pi / 180.0;
        const viewport_height = 2.0 * std.math.tan(vfov_rad / 2);
        const viewport_width = aspect_ratio * viewport_height;

        const w = vec3fNormalize(origin - look_pos);
        const u = vec3fNormalize(vec3fCross(up, w));
        const v = vec3fCross(w, u);

        var cam: Camera = undefined;
        cam.origin = origin;
        cam.horizontal = vec3fSplat(focus_dist * viewport_width) * u;
        cam.vertical = vec3fSplat(focus_dist * viewport_height) * v;
        cam.lower_left_corner = origin - cam.horizontal / vec3fSplat(2) - cam.vertical / vec3fSplat(2) - vec3fSplat(focus_dist) * w;
        cam.lens_radius = aperture / 2;
        cam.w = w;
        cam.u = u;
        cam.v = v;
        return cam;
    }

    fn getRay(self: Camera, s: f32, t: f32) Ray {
        const rd = vec3fSplat(self.lens_radius) * vec3fRandUnitDisc();
        const offset = self.u * vec3fSplat(rd[0]) + self.v * vec3fSplat(rd[1]);
        return Ray.new(
            self.origin + offset,
            self.lower_left_corner + vec3fSplat(s) * self.horizontal + vec3fSplat(t) * self.vertical - self.origin - offset,
        );
    }
};

// ===========================================================================
// Vector math
// ===========================================================================

/// Scalar to vector
fn vec3fSplat(x: anytype) Vec3f {
    return @splat(3, @as(f32, x));
}

fn vec3f(x: f32, y: f32, z: f32) Vec3f {
    return .{ x, y, z };
}

fn vec3fNorm(v: Vec3f) f32 {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

fn vec3fNormalize(v: Vec3f) Vec3f {
    return v / vec3fSplat(vec3fNorm(v));
}

fn vec3fDot(a: Vec3f, b: Vec3f) f32 {
    return @reduce(.Add, a * b);
}

fn vec3fCross(a: Vec3f, b: Vec3f) Vec3f {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn vec3fRand(min: f32, max: f32) Vec3f {
    return .{ randf32(min, max), randf32(min, max), randf32(min, max) };
}

fn vec3fRandUnitDisc() Vec3f {
    while (true) {
        const p = vec3f(randf32(-1, 1), randf32(-1, 1), 0);
        if (vec3fDot(p, p) >= 1) continue;
        return p;
    }
}

fn vec3fRandUnitSphere() Vec3f {
    while (true) {
        const p = vec3fRand(-1, 1);
        if (vec3fDot(p, p) >= 1) continue;
        return p;
    }
}

fn vec3fNearZero(v: Vec3f) bool {
    const s = std.math.f32_epsilon;
    return std.math.fabs(v[0]) < s and
        std.math.fabs(v[1]) < s and
        std.math.fabs(v[2]) < s;
}

fn vec3fReflect(v: Vec3f, n: Vec3f) Vec3f {
    return v - vec3fSplat(2) * vec3fSplat(vec3fDot(v, n)) * n;
}

/// Refracts incident ray `uv` where `n` is the normal and
/// `eta_ratio` is the ratio of the refractive indices `eta/eta'`
/// (eta' is the index of the material on the refracted side)
fn vec3fRefract(uv: Vec3f, n: Vec3f, eta_ratio: f32) Vec3f {
    const cos = std.math.min(-vec3fDot(uv, n), 1.0);
    const out_perp = vec3fSplat(eta_ratio) * (uv + vec3fSplat(cos) * n);
    const out_parallel = vec3fSplat(-sqrt(std.math.fabs(1.0 - vec3fDot(out_perp, out_perp)))) * n;
    return out_perp + out_parallel;
}

// ===========================================================================
// Rendering code
// ===========================================================================

fn rayColor(ray: Ray, world: World, depth: u32) Vec3f {
    if (depth == 0) return vec3fSplat(0);

    if (world.hit(ray, .{ .t_min = 0.001, .t_max = std.math.f32_max })) |rec| {
        var attenuation: Vec3f = undefined;
        if (rec.material.scatter(ray, rec, &attenuation)) |scattered| {
            return attenuation * rayColor(scattered, world, depth - 1);
        } else return vec3fSplat(0);
    }

    const t = 0.5 * (vec3fNormalize(ray.direction)[1] + 1.0);
    const c = vec3fSplat(1.0 - t) * vec3f(1.0, 1.0, 1.0) + vec3fSplat(t) * vec3f(0.5, 0.7, 1.0);
    return c;
}

const RenderArgs = struct {
    x_off: usize,
    y_off: usize,
    x_size: usize,
    y_size: usize,
};

const WorkerContext = struct {
    allocator: std.mem.Allocator,
    world: World,
    camera: Camera,
    image_width: usize,
    image_height: usize,
    samples_per_pixel: u32,
    max_depth: u32,
    frame_buffer: []Vec3f,
    queue: *WorkQueue,
};

fn renderThread(ctx: WorkerContext) void {
    while (true) {
        if (ctx.queue.get()) |node| {
            defer ctx.allocator.destroy(node);
            const work = node.data;
            var j: usize = work.y_off;
            while (j < work.y_off + work.y_size) : (j += 1) {
                var i: usize = work.x_off;
                while (i < work.x_off + work.x_size) : (i += 1) {
                    var pixel_color = vec3fSplat(0);
                    var samples: u32 = 0;
                    while (samples < ctx.samples_per_pixel) : (samples += 1) {
                        const s = (@intToFloat(f32, i) + randf32(0.0, 1.0)) / @intToFloat(f32, ctx.image_width - 1);
                        const t = (@intToFloat(f32, j) + randf32(0.0, 1.0)) / @intToFloat(f32, ctx.image_height - 1);
                        const r = ctx.camera.getRay(s, t);
                        pixel_color += rayColor(r, ctx.world, ctx.max_depth);
                    }
                    ctx.frame_buffer[i + (ctx.image_height - 1 - j) * ctx.image_width] = pixel_color;
                }
            }
        } else return;
    }
}

pub fn main() anyerror!void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var timer = try std.time.Timer.start();

    // Init PRNG
    rand = std.rand.DefaultPrng.init(@intCast(u64, std.time.timestamp()));

    const image_width = 1200;
    const image_height = 900;
    const samples_per_pixel = 10;
    const max_depth = 100;
    const aspect_ratio = @intToFloat(f32, image_width) / @intToFloat(f32, image_height);

    var world = try World.init(std.testing.allocator);
    defer world.deinit();
    try world.add(Sphere{
        .center = vec3f(0, -1000, -0.5),
        .radius = 1000,
        .material = Material.lambertian(vec3f(0.8, 0.8, 0.0)),
    });

    var a: f32 = -11;
    while (a < 11) : (a += 1) {
        var b: f32 = -11;
        while (b < 11) : (b += 1) {
            const radius = randf32(0.1, 0.25);
            const center = vec3f(a + randf32(0, 0.9), radius, b + randf32(0, 0.9));
            const choose_material = randf32(0, 1);
            const material = if (choose_material < 0.60)
                Material.lambertian(vec3f(randf32(0, 1), randf32(0, 1), randf32(0, 1)))
            else if (choose_material < 0.95)
                Material.metal(vec3f(randf32(0, 1), randf32(0, 1), randf32(0, 1)), randf32(0, 1))
            else
                Material.dielectric(1.5);

            try world.add(Sphere{
                .center = center,
                .radius = radius,
                .material = material,
            });
        }
    }
    try world.add(Sphere{
        .center = vec3f(0.0, 1.0, 0.0),
        .radius = 1.0,
        .material = Material.dielectric(1.5),
    });
    try world.add(Sphere{
        .center = vec3f(-4.0, 1.0, 0.0),
        .radius = 1.0,
        .material = Material.lambertian(vec3f(0.4, 0.1, 0.8)),
    });
    try world.add(Sphere{
        .center = vec3f(4, 1.0, 0.0),
        .radius = 1.0,
        .material = Material.metal(vec3f(0.9, 0.3, 0.3), 0.1),
    });

    // Camera
    const origin = vec3f(13, 2, 3);
    const look_pos = vec3f(0, 0, 0);
    const camera = Camera.init(
        origin,
        look_pos,
        vec3f(0, 1, 0), // up
        20,
        aspect_ratio,
        0.1, // aperture
        10.0, // distance to focus
    );

    // Render to buffer
    var frame_buffer = try allocator.alloc(Vec3f, image_width * image_height);
    defer allocator.free(frame_buffer);

    // Write to file
    var file = try std.fs.cwd().createFile("output.ppm", .{});
    defer file.close();
    var buf_writer = std.io.bufferedWriter(file.writer());
    defer buf_writer.flush() catch {};
    var writer = buf_writer.writer();
    try writer.print("P3\n{} {}\n255\n", .{ image_width, image_height });

    // Tile up
    var queue = std.atomic.Queue(RenderArgs).init();

    const ctx = WorkerContext{
        .allocator = allocator,
        .world = world,
        .camera = camera,
        .image_width = image_width,
        .image_height = image_height,
        .samples_per_pixel = samples_per_pixel,
        .max_depth = max_depth,
        .queue = &queue,
        .frame_buffer = frame_buffer,
    };

    const tile_size = 256;
    var y_off: usize = 0;
    while (y_off < image_height) : (y_off += tile_size) {
        var x_off: usize = 0;
        while (x_off < image_width) : (x_off += tile_size) {
            const x_size = if (x_off + tile_size < image_width) tile_size else image_width - x_off;
            const y_size = if (y_off + tile_size < image_height) tile_size else image_height - y_off;
            var node = try allocator.create(WorkQueue.Node);
            node.data = .{
                .x_off = x_off,
                .y_off = y_off,
                .x_size = x_size,
                .y_size = y_size,
            };
            queue.put(node);
        }
    }

    const num_threads = try std.Thread.getCpuCount();
    std.log.info("Using {} threads.", .{num_threads});
    var threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, renderThread, .{ctx});
    for (threads) |t| t.join();

    for (frame_buffer) |p| {
        const scaled = p * vec3fSplat(1.0 / @intToFloat(f32, samples_per_pixel));
        // Gmma correction; we use gamma = 2
        const corrected = vec3f(sqrt(scaled[0]), sqrt(scaled[1]), sqrt(scaled[2]));
        const quantized = corrected * vec3fSplat(255.9999);
        try writer.print("{} {} {}\n", .{
            @floatToInt(u8, std.math.clamp(quantized[0], 0.0, 255.9999)),
            @floatToInt(u8, std.math.clamp(quantized[1], 0.0, 255.9999)),
            @floatToInt(u8, std.math.clamp(quantized[2], 0.0, 255.9999)),
        });
    }
    std.log.info("Took {} seconds.", .{timer.read() / std.time.ns_per_s});
}
