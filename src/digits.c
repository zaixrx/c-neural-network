#include "nn.h"
#include "nn_math.h"
#include <math.h>
#include <stdint.h>
#include <raylib.h>
#include <stdlib.h>

#define SCALE 20
#define GRID 28
#define WIN_W (GRID * SCALE)
#define WIN_H (GRID * SCALE)
#define FPS 10

// Apply a circular brush with linear falloff
void ApplyBrush(vec_t image, int cx, int cy, int radius) {
	for (int y = cy - radius; y <= cy + radius; ++y) {
		for (int x = cx - radius; x <= cx + radius; ++x) {
			if (x < 0 || x >= GRID || y < 0 || y >= GRID) continue;

    	    	    	float dx = x - cx;
    	    	    	float dy = y - cy;
    	    	    	float dist = sqrtf(dx*dx + dy*dy);

    	    	    	if (dist > radius) continue;

    	    	    	float strength = 1.0f - (dist / radius);
    	    	    	uint8_t addVal = (uint8_t)(strength * 255);

    	    	    	int newVal = image[y*GRID+x] + addVal;
    	    	    	if (newVal > 255) newVal = 255;
    	    	    	image[y*GRID+x] = newVal;
    	    	}
    	}
}

int main(void) {
	Network net = {0};
	assert(network_import(&net, "nn.data") == NN_CODE_SUCCESS);

	vec_t image = vec_new(GRID * GRID);

	InitWindow(WIN_W, WIN_H, "digits example");
    	SetTargetFPS(FPS);

    	Vector2 prevMouse = (Vector2){ -1, -1 };
    	int brushRadius = 2;

    	while (!WindowShouldClose()) {
		Vector2 mouse = GetMousePosition();

		if (IsKeyDown(KEY_SPACE)) {
			memset(image, 0, GRID * GRID);
		}

    	    	if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
			int i = (int)mouse.x / SCALE;
    	    	    	int j = (int)mouse.y / SCALE;

    	    	    	if (i >= 0 && i < GRID && j >= 0 && j < GRID) {
    	    	    		if (prevMouse.x >= 0 && prevMouse.y >= 0) {
    	    	    	    	    	int x0 = (int)(prevMouse.x / SCALE);
    	    	    	    	    	int y0 = (int)(prevMouse.y / SCALE);

					if (x0 >= GRID) x0 = GRID;
					if (y0 >= GRID) y0 = GRID;

    	    	    	    	    	int dx = abs(i - x0), sx = x0 < i ? 1 : -1;
                    			int dy = -abs(j - y0), sy = y0 < j ? 1 : -1;
					int err = (dx + dy), e2;

					while (1) {
    	    	    	    	    	    	ApplyBrush(image, x0, y0, brushRadius);
    	    	    	    	    	    	if (x0 == i && y0 == j) break;
    	    	    	    	    	    	e2 = 2 * err;
    	    	    	    	    	    	if (e2 >= dy) { err += dy; x0 += sx; }
    	    	    	    	    	    	if (dx >= e2) { err += dx; y0 += sy; }
    	    	    	    	    	}
    	    	    	    	} else {
			    		ApplyBrush(image, i, j, brushRadius);
    	    	    	    	}
    	    	    	}

			double *out = network_feedforward(&net, image);
			size_t max = 0;
			for (size_t i = 1; i < arrlen(out); ++i) if (out[i] > out[max]) max = i;
			printf("%zu with %d percent\n", max, (int)(out[max]*100));
			// vec_print(out);
			vec_destroy(out);
    	    	} else {
    	    	    	prevMouse.x = -1;
    	    	    	prevMouse.y = -1;
    	    	}
    	    	prevMouse = mouse;

    	    	BeginDrawing();
    	    	ClearBackground(BLACK);
    	    	for (int y = 0; y < GRID; ++y) {
    	    	    	for (int x = 0; x < GRID; ++x) {
    	    	       		if (!image[y*GRID+x]) continue;

    	    	        	Color c = (Color){255, 255, 255, image[y*GRID+x]};
    	    	        	DrawRectangle(x * SCALE, y * SCALE, SCALE, SCALE, c);
    	    	    	}
    	    	}
    	    	EndDrawing();
    	}

    	CloseWindow();
    	return 0;
}
