main: src/main.c src/nn.c
	cc -o main src/main.c src/nn.c -I./include -lm
