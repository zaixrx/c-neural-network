#include <math.h>
#include <raylib.h>
#include <raymath.h>

#define WIN_W 800
#define WIN_H 600
#define FPS 60

#define MIN_ZOOM 0.75
#define MAX_ZOOM 2
#define UNIT_SIZE 100
#define MAX_POINTS 10

typedef float(*Func1D)(float);

int IntegerPart(float x) {
	if (x >= 0) return (int)x;
	return (int)x-1;
}

Vector2 GetCamOffset(Camera2D camera) {
	return (Vector2){
		.x = IntegerPart(camera.target.x/UNIT_SIZE)*UNIT_SIZE,
		.y = IntegerPart(camera.target.y/UNIT_SIZE)*UNIT_SIZE,
	};
}

void DrawCartesianGraph(Camera2D camera, float thick, Color color) {
	DrawLineEx(
		(Vector2){ WIN_W/2.0F, camera.target.y },
		(Vector2){ WIN_W/2.0F, camera.target.y+WIN_H },
		thick,
		color
	);
	DrawLineEx(
		(Vector2){ camera.target.x, WIN_H/2.0F },
		(Vector2){ camera.target.x+WIN_W, WIN_H/2.0F },
		thick,
		color
	);

	color.a /= 10;
	Vector2 offset = GetCamOffset(camera);
	for (int x = 0; x <= WIN_W/UNIT_SIZE; ++x) {
		DrawLineEx(
			(Vector2){ offset.x+UNIT_SIZE*x, camera.target.y       },
			(Vector2){ offset.x+UNIT_SIZE*x, camera.target.y+WIN_H },
			thick,
			color
		);
	}
	for (int y = 0; y <= WIN_H/UNIT_SIZE; ++y) {
		DrawLineEx(
			(Vector2){ camera.target.x      , offset.y+UNIT_SIZE*y },
			(Vector2){ camera.target.x+WIN_W, offset.y+UNIT_SIZE*y },
			thick,
			color
		);
	}
}

static inline float sign(float x) {
	return x >= 0 ? 1 : -1;
}

static inline float ToCartesianX(float x) {
	return (x-(WIN_W>>1))/UNIT_SIZE;
}

static inline float ToCartesianY(float y) {
	return -(y-(WIN_H>>1))/UNIT_SIZE;
}

static inline float ToWorldY(float y) {
	return WIN_H-(y*UNIT_SIZE+(WIN_H>>1));
}

void GraphFunc1D(Func1D fn, float step, Camera2D camera, float thick, Color color) {
	float win2 = WIN_W/2.0F;
	for (float x = camera.target.x; x <= camera.target.x+WIN_W; x += step) {
		Vector2 start = { x, ToWorldY(fn(ToCartesianX(x))) };
		Vector2 end   = { x+step, ToWorldY(fn(ToCartesianX(x+step))) };
		DrawLineEx(start, end, thick, color);
	}
}

void DrawPoint(Vector2 pos, float radius, Color color) {
	DrawCircle(pos.x, pos.y, radius, color);
}

int main(void) {
	int points_n = 0;
	int points_i = 0;
	Vector2 points[MAX_POINTS];
	InitWindow(WIN_W, WIN_H, "grapher");
	Camera2D camera = {0};
	camera.zoom = 1.0F;
	Vector2 prev_mouse = { -1, -1 };
	while (!WindowShouldClose()) {
		// TODO: zoom
		// camera.zoom = Clamp(camera.zoom+GetMouseWheelMove()/10.0F, MIN_ZOOM, MAX_ZOOM);
		Vector2 mouse = GetMousePosition();
		if (IsKeyDown(KEY_SPACE)) {
			camera.target = (Vector2){ 0, 0 };
		}
		if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
			points[points_i] = Vector2Add(camera.target, mouse);
			points_i = (points_i+1) % MAX_POINTS;
			if (points_n < MAX_POINTS) ++points_n;
		}
		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
			if (prev_mouse.x >= 0 && prev_mouse.y >= 0) {
				Vector2 delta = Vector2Subtract(mouse, prev_mouse);
				camera.target = Vector2Subtract(camera.target, delta);
			}
			prev_mouse = mouse;
		} else {
			prev_mouse = (Vector2){ -1, -1 };
		}
		BeginDrawing();
        		ClearBackground(BLACK);
			BeginMode2D(camera);
			DrawCartesianGraph(camera, 2.0F, WHITE);
			GraphFunc1D(expf, 0.1F, camera, 2.0F, BLUE);
			for (int i = 0; i < points_n; ++i) {
				DrawPoint(points[i], 10, RED);
			}
			EndMode2D();
        	EndDrawing();
    	}

    	CloseWindow();

    	return 0;
}
