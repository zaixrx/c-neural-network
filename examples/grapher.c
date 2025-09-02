#include <raylib.h>
#include <raymath.h>

#define WIN_W 800
#define WIN_H 600
#define FPS 60

int IntegerPart(float x) {
	if (x >= 0) return (int)x;
	return (int)x-1;
}

void DrawCartesianGraph(Camera2D camera, int unit_size, float thickness, Color color) {
	DrawLineEx(
		(Vector2){ WIN_W/2.0F, camera.target.y },
		(Vector2){ WIN_W/2.0F, camera.target.y+WIN_H },
		2.0F,
		color
	);
	DrawLineEx(
		(Vector2){ camera.target.x, WIN_H/2.0F },
		(Vector2){ camera.target.x+WIN_W, WIN_H/2.0F },
		2.0F,
		color
	);

	color.a /= 4;
	float left_most = IntegerPart(camera.target.x/unit_size)*unit_size;
	float top_most  = IntegerPart(camera.target.y/unit_size)*unit_size;
	for (int x = 0; x <= WIN_W/unit_size; ++x) {
		DrawLineEx(
			(Vector2){ left_most+unit_size*x, camera.target.y },
			(Vector2){ left_most+unit_size*x, camera.target.y+WIN_H },
			2.0F,
			color
		);
	}
	for (int y = 0; y <= WIN_H/unit_size; ++y) {
		DrawLineEx(
			(Vector2){ camera.target.x, top_most+unit_size*y },
			(Vector2){ camera.target.x+WIN_W, top_most+unit_size*y },
			2.0F,
			color
		);
	}
}

int main(void) {
	InitWindow(WIN_W, WIN_H, "grapher");

	Camera2D camera = {0};
	camera.zoom = 1.0F;
	Vector2 prev_mouse = { -1, -1 };
	while (!WindowShouldClose()) {
		if (IsKeyDown(KEY_SPACE)) {
			camera.target = (Vector2){ 0, 0 };
		}
		if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
			Vector2 mouse = GetMousePosition();
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
			DrawCartesianGraph(camera, 100.0F, 25.0F, WHITE);
			EndMode2D();
        	EndDrawing();
    	}

    	CloseWindow();

    	return 0;
}
