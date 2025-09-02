#include "nn.h"
#include <raylib.h>
#include <raymath.h>
#include <stdio.h>

#define WIN_W 1200
#define WIN_H 800
#define FPS 120

#define MIN_ZOOM 0.75
#define MAX_ZOOM 2
#define UNIT_SIZE 100
#define MAX_POINTS 10

Network net = {0};

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

float nn_fn(float x) {
	vec_t in = NULL;
	arrpush(in, x);
	vec_t out = network_feedforward(&net, in);
	float val = out[0];
	arrfree(in);
	arrfree(out);
	return val;
}

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

DataEntry *InitNetwork(Vector2 *points, unsigned int points_n) {
	DataEntry *training_set = NULL;
	for (int i = -100; i <= 100; ++i) {
		DataEntry entry = {0};
		arrpush(entry.x, i);
		arrpush(entry.y, sigmoid(i));
		arrpush(training_set, entry);
	}
	net.activation = NN_SIGMOID;
	network_create(&net, 2, 1, 1);
	return training_set;
}

static char text_buffer[128];
void NetworkLearn(Camera2D camera, DataEntry *training_set, float learning_rate) {
	static unsigned int e = 0;
	network_SGD(&net, arrlen(training_set), learning_rate, training_set);
	double total_cost = 0.0F;
	for (int i = 0; i < arrlen(training_set); ++i) total_cost += network_cost(&net, training_set[i]);
	GraphFunc1D(nn_fn, 0.1F, camera, 2.0F, BLUE);
	sprintf(text_buffer, "epoch: %d\ncost: %f", e, total_cost);
	DrawText(text_buffer, 0, 0, 25, WHITE);
	++e;
}

int main(void) {
	int points_n = 0;
	int points_i = 0;
	Vector2 points[MAX_POINTS];

	InitWindow(WIN_W, WIN_H, "grapher");
	Camera2D camera = {0};
	camera.zoom = 1.0F;
	Vector2 prev_mouse = { -1, -1 };
	DataEntry *training_set = NULL;
	while (!WindowShouldClose()) {
		Vector2 mouse = GetMousePosition();
		if (IsKeyDown(KEY_SPACE)) {
			camera.target = (Vector2){ 0, 0 };
		}
		if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) && !training_set) {
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
		if (IsKeyDown(KEY_ENTER)) {
			training_set = InitNetwork(points, points_n);
		}
		BeginDrawing();
        		ClearBackground(BLACK);
			BeginMode2D(camera);
				DrawCartesianGraph(camera, 2.0F, WHITE);
				for (int i = 0; i < points_n; ++i) {
					DrawPoint(points[i], 10, RED);
				}
				if (training_set) {
					NetworkLearn(camera, training_set, 1e-2);
				}
			EndMode2D();
        	EndDrawing();
    	}
    	CloseWindow();
	network_destroy(&net);

    	return 0;
}
